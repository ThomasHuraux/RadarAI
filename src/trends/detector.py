import json
import numpy as np
from datetime import date, timedelta
from collections import defaultdict
from src.storage import db
from src.nlp.keywords import extract_keywords, get_cluster_name
from src.nlp import llm_namer, ollama_client


def _cluster_centroid(arts: list[dict]) -> np.ndarray | None:
    vecs = []
    for a in arts:
        emb = a.get("embedding")
        if isinstance(emb, str):
            emb = json.loads(emb)
        if emb:
            vecs.append(np.array(emb, dtype=np.float32))
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def _centroid_title(arts: list[dict]) -> str:
    """Article le plus proche du centroïde — utilisé en fallback si les mots-clés sont vides."""
    vecs = [(a, np.array(json.loads(a["embedding"]) if isinstance(a.get("embedding"), str) else a.get("embedding", []), dtype=np.float32))
            for a in arts if a.get("embedding")]
    if not vecs:
        return ""
    centroid = np.mean([v for _, v in vecs], axis=0)
    closest = min(vecs, key=lambda x: np.linalg.norm(x[1] - centroid))
    return closest[0]["title"]


# Similarité cosinus minimale entre centroïdes pour considérer que deux clusters
# de jours différents portent le même sujet. Les embeddings sont stables d'un jour
# à l'autre (contrairement aux cluster_id) donc ce signal est bien plus fiable que
# le recouvrement de mots-clés — mais il faut un centroïde des deux côtés (clusters
# persistés avant l'ajout de la colonne `centroid` n'en ont pas), d'où le fallback.
MIN_CENTROID_MATCH = 0.72

# Fallback historique quand un centroïde manque (migration, cluster à 0 embedding...).
# Coefficient de chevauchement (overlap / min(len_a, len_b)) : environ "2 mots-clés
# communs sur 8" suffisent à considérer que c'est le même sujet. Ajustable.
MIN_KEYWORD_OVERLAP = 0.25


def _match_yesterday_counts(
    today_keywords: dict[int, list[str]],
    today_centroids: dict[int, np.ndarray | None],
    yesterday_clusters: list[dict],
) -> dict[int, int]:
    """
    Mappe chaque cluster d'aujourd'hui vers son équivalent d'hier.

    Les cluster_id sont réassignés à zéro chaque jour (KMeans/HDBSCAN repart de zéro),
    donc comparer les ID directement est faux. Signal primaire : similarité cosinus
    entre centroïdes d'embedding (stables d'un jour à l'autre, contrairement aux ID).
    Fallback : recouvrement de mots-clés, utilisé seulement quand un centroïde manque
    d'un côté ou de l'autre.
    """
    result: dict[int, int] = {}
    for cid, keywords in today_keywords.items():
        today_set = set(keywords)
        today_centroid = today_centroids.get(cid)

        # Passe 1 : matching par centroïde, uniquement contre les clusters d'hier
        # qui en ont un (les deux échelles de score — cosinus vs overlap de mots-clés
        # — ne sont pas comparables, donc on ne les mélange jamais dans une même passe).
        best_sim = 0.0
        best_count_by_centroid = 0
        if today_centroid is not None:
            for yc in yesterday_clusters:
                yest_centroid = yc.get("centroid")
                if yest_centroid is None:
                    continue
                sim = float(
                    np.dot(today_centroid, yest_centroid)
                    / (np.linalg.norm(today_centroid) * np.linalg.norm(yest_centroid) + 1e-9)
                )
                if sim > best_sim:
                    best_sim = sim
                    best_count_by_centroid = yc.get("article_count", 0)

        if best_sim >= MIN_CENTROID_MATCH:
            result[cid] = best_count_by_centroid
            continue

        # Passe 2 : fallback par recouvrement de mots-clés — pour les clusters d'hier
        # persistés sans centroïde, ou quand aucun match centroïde n'a dépassé le seuil.
        best_overlap = 0.0
        best_count = 0
        if today_set:
            for yc in yesterday_clusters:
                yest_set = set(yc.get("keywords", []))
                if not yest_set:
                    continue
                overlap = len(today_set & yest_set) / min(len(today_set), len(yest_set))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_count = yc.get("article_count", 0)

        result[cid] = best_count if best_overlap >= MIN_KEYWORD_OVERLAP else 0

    return result


# Familles de sources indépendantes. Le but du score de tendance n'est PAS de compter
# des articles mais de détecter un sujet couvert par plusieurs voix INDÉPENDANTES —
# "tout le monde en parle", pas "cette API a retourné 15 résultats sur ce thème".
# `arxiv` + `huggingface_papers` ne comptent qu'une seule famille : HuggingFace Daily
# Papers republie littéralement les mêmes papers arXiv (même contenu, pas une seconde
# opinion). `semanticscholar` fait la même chose (agrégateur académique), d'où le
# regroupement des trois dans "academic" plutôt que de les compter comme 3 sources
# distinctes — c'était l'angle mort de l'ancien score (count*0.6 + growth*0.4), qui
# faisait gagner un cluster de 13 papers académiques sur un cluster de 5 articles
# provenant réellement de 2 mondes différents (presse + outils).
SOURCE_FAMILIES = {
    "arxiv": "academic",
    "huggingface_papers": "academic",
    "semanticscholar": "academic",
    "techcrunch": "press",
    "venturebeat": "press",
    "mit_tech_review": "press",
    "the_verge_ai": "press",
    "wired_ai": "press",
    "reddit_ml": "social",
    "reddit_localllama": "social",
    "reddit_artificial": "social",
    "hackernews": "social",
    "github_releases": "tooling",
    "openai_blog": "official",
    "anthropic_blog": "official",
}


def _source_family_count(sources: list[str]) -> int:
    return len({SOURCE_FAMILIES.get(s, s) for s in sources})


# Poids dominant : chaque famille indépendante supplémentaire vaut largement plus que
# du volume brut, pour qu'un sujet couvert par 3 mondes différents (presse, réseaux,
# officiel) batte toujours un cluster de 50 articles issus d'une seule famille.
FAMILY_WEIGHT = 6.0
VOLUME_WEIGHT = 0.3
GROWTH_WEIGHT = 0.2

# Un cluster confiné à une seule famille de sources n'est jamais un "sujet du jour"
# corroboré — que ce soit 50 papers académiques ou 20 posts du même sub Reddit. Il
# reste affichable (un papier vraiment isolé peut être important), mais fortement
# relégué plutôt que de dominer le classement sur le seul volume.
SINGLE_FAMILY_PENALTY = 0.25


def compute_trend_score(count_today: int, count_yesterday: int, sources: list[str]) -> float:
    """
    Score de tendance : la diversité de familles de sources INDÉPENDANTES domine ;
    volume et croissance ne servent qu'à départager des sujets à diversité égale.
    """
    family_count = _source_family_count(sources)
    growth_rate = (count_today - count_yesterday) / max(1, count_yesterday)
    score = family_count * FAMILY_WEIGHT + count_today * VOLUME_WEIGHT + growth_rate * GROWTH_WEIGHT
    if family_count <= 1:
        score *= SINGLE_FAMILY_PENALTY
    return round(score, 4)


def build_clusters(articles: list[dict], target_date: str, cohesion_threshold: float | None = None) -> list[dict]:
    if cohesion_threshold is None:
        from src.nlp.clusterer import MIN_CLUSTER_FIT
        cohesion_threshold = MIN_CLUSTER_FIT

    yesterday = (date.fromisoformat(target_date) - timedelta(days=1)).isoformat()
    yesterday_clusters = db.get_clusters_by_date(yesterday)

    # Un seul health check par run (pas par cluster) : évite 15-20 aller-retours
    # HTTP redondants si Ollama est simplement indisponible ce coup-ci.
    ollama_up = ollama_client.is_available()

    cluster_articles: dict[int, list[dict]] = defaultdict(list)
    for a in articles:
        cid = a.get("cluster_id", -1)
        if cid == -1:
            continue
        cluster_articles[cid].append(a)

    # Corpus global du jour pour l'IDF : tous les articles, pas seulement ceux du cluster.
    # Permet à TF-IDF de pénaliser "large language model" qui apparaît partout.
    all_texts = [f"{a['title']} {a.get('content', '')[:200]}" for a in articles]

    # Mots-clés et centroïdes calculés en premier pour chaque cluster, nécessaires
    # au matching cross-jour (centroïde en priorité, mots-clés en fallback).
    today_keywords: dict[int, list[str]] = {}
    today_centroids: dict[int, np.ndarray | None] = {}
    cluster_names: dict[int, str] = {}
    cluster_labeling_method: dict[int, str] = {}
    for cid, arts in cluster_articles.items():
        texts = [f"{a['title']} {a.get('content', '')[:200]}" for a in arts]
        keywords = extract_keywords(texts, corpus=all_texts)
        today_keywords[cid] = keywords
        today_centroids[cid] = _cluster_centroid(arts)

        # Mots-clés en priorité : ils sont extraits de TOUS les articles du cluster
        # et représentent le sujet commun. Le titre centroïde est un seul article —
        # il peut être trompeur si le cluster est hétérogène. Sert de fallback si
        # le titrage LLM est indisponible ou échoue pour ce cluster.
        fallback_name = get_cluster_name(keywords) or _centroid_title(arts)
        name, method = llm_namer.generate_cluster_title(arts, fallback_name, ollama_up)
        cluster_names[cid] = name
        cluster_labeling_method[cid] = method

    # Matching cross-jour par recouvrement de mots-clés (les ID et les embeddings
    # ne sont pas stables/comparables d'un jour à l'autre, cf. _match_yesterday_counts)
    yesterday_counts = _match_yesterday_counts(today_keywords, today_centroids, yesterday_clusters)

    duplicate_sources = db.get_duplicate_sources_by_date(target_date)

    clusters = []
    for cid, arts in cluster_articles.items():
        keywords = today_keywords[cid]
        name = cluster_names[cid]

        count_today = len(arts)
        count_yest = yesterday_counts.get(cid, 0)

        top_arts = sorted(arts, key=lambda x: len(x.get("content", "")), reverse=True)[:3]
        top_titles = [{"title": a["title"], "url": a.get("url", ""), "source": a.get("source", "")} for a in top_arts]

        # Cohésion : similarité moyenne des membres à leur centroïde (cf. clusterer.py).
        # None pour les articles dont le cluster n'a jamais été "gaté" (cas <5 articles/jour).
        fits = [a["cluster_fit"] for a in arts if a.get("cluster_fit") is not None]
        cohesion = round(float(np.mean(fits)), 4) if fits else 0.0

        # Diversité des sources : réunit les sources des articles du cluster ET celles
        # des doublons fusionnés (exclus du clustering mais toujours une corroboration
        # réelle — cf. src/processor/deduplicator.py).
        sources = {a.get("source", "") for a in arts}
        for a in arts:
            sources.update(duplicate_sources.get(a["id"], []))
        sources.discard("")
        sources = sorted(sources)

        score = compute_trend_score(count_today, count_yest, sources)

        centroid = today_centroids.get(cid)

        clusters.append({
            "id": cid,
            "name": name,
            "centroid": centroid.tolist() if centroid is not None else None,
            "keywords": keywords,
            "article_count": count_today,
            "yesterday_count": count_yest,
            "trend_score": score,
            "top_titles": top_titles,
            "articles": arts,
            "cohesion": cohesion,
            "sources": sources,
            "source_count": len(sources),
            "low_confidence": cohesion < cohesion_threshold,
            "labeling_method": cluster_labeling_method[cid],
        })

    clusters.sort(key=lambda c: -c["trend_score"])
    return clusters
