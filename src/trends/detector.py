import json
import numpy as np
from datetime import date, timedelta
from collections import defaultdict
from src.storage import db
from src.nlp.keywords import extract_keywords, get_cluster_name


def _central_title(arts: list[dict]) -> str:
    """
    Retourne le titre de l'article le plus proche du centroïde du cluster.

    Idée : le centroïde est la "moyenne" de tous les vecteurs du cluster.
    L'article le plus proche de cette moyenne est le plus "représentatif"
    du sujet commun — c'est lui qui sert de titre au cluster.

    C'est plus fiable qu'un LLM (0€) et plus lisible que les mots-clés TF-IDF.
    """
    vecs = []
    for a in arts:
        emb = a.get("embedding")
        if isinstance(emb, str):
            emb = json.loads(emb)
        if emb:
            vecs.append((a, np.array(emb, dtype=np.float32)))

    if not vecs:
        return ""

    # Centroïde = moyenne arithmétique des vecteurs
    centroid = np.mean([v for _, v in vecs], axis=0)
    # Article le plus proche = distance euclidienne minimale au centroïde
    closest = min(vecs, key=lambda x: np.linalg.norm(x[1] - centroid))
    return closest[0]["title"]


def compute_trend_score(count_today: int, count_yesterday: int) -> float:
    """
    Score de tendance : combine volume absolu et croissance relative.

    Formule : count * 0.6 + growth_rate * 0.4

    - count * 0.6 : un sujet avec 50 articles est intrinsèquement plus important
      qu'un sujet avec 3 articles, même si les deux ont doublé.
    - growth_rate * 0.4 : un sujet qui explose aujourd'hui remonte dans le classement
      même s'il était petit hier.

    Exemple :
      Sujet A : 50 articles aujourd'hui, 40 hier → score = 50*0.6 + 0.25*0.4 = 30.1
      Sujet B :  5 articles aujourd'hui,  1 hier → score =  5*0.6 + 4.0*0.4  =  4.6
    """
    growth_rate = (count_today - count_yesterday) / max(1, count_yesterday)
    return round(count_today * 0.6 + growth_rate * 0.4, 4)


def build_clusters(articles: list[dict], target_date: str) -> list[dict]:
    """
    Construit les objets cluster enrichis à partir des articles déjà clusterisés.

    Ce module est appelé APRÈS le clustering (cluster_id est déjà assigné).
    Son rôle est d'enrichir chaque cluster avec :
      - un nom (titre centroïde ou mots-clés)
      - un trend_score (voir compute_trend_score)
      - les articles "top" (les plus longs = les mieux documentés)
      - le count J-1 pour calculer la croissance
    """
    # On charge les articles d'hier pour calculer la croissance
    yesterday = (date.fromisoformat(target_date) - timedelta(days=1)).isoformat()
    yesterday_articles = db.get_articles_by_date(yesterday)

    # Compte des articles par cluster_id pour J-1
    yesterday_counts: dict[int, int] = defaultdict(int)
    for a in yesterday_articles:
        yesterday_counts[a["cluster_id"]] += 1

    # Groupement des articles d'aujourd'hui par cluster_id
    cluster_articles: dict[int, list[dict]] = defaultdict(list)
    for a in articles:
        cid = a.get("cluster_id", -1)
        if cid == -1:
            continue  # articles non assignés (bruit HDBSCAN)
        cluster_articles[cid].append(a)

    clusters = []
    for cid, arts in cluster_articles.items():
        # Extraction des mots-clés du cluster à partir de titre + début de contenu
        texts = [f"{a['title']} {a.get('content', '')[:200]}" for a in arts]
        keywords = extract_keywords(texts)

        # Titre = article centroïde, ou mots-clés si pas d'embeddings disponibles
        name = _central_title(arts) or get_cluster_name(keywords)

        count_today = len(arts)
        count_yest = yesterday_counts.get(cid, 0)
        score = compute_trend_score(count_today, count_yest)

        # Top 3 articles = les plus longs (contenu le plus riche)
        top_arts = sorted(arts, key=lambda x: len(x.get("content", "")), reverse=True)[:3]
        top_titles = [{"title": a["title"], "url": a.get("url", ""), "source": a.get("source", "")} for a in top_arts]

        clusters.append({
            "id": cid,
            "name": name,
            "keywords": keywords,
            "article_count": count_today,
            "yesterday_count": count_yest,
            "trend_score": score,
            "top_titles": top_titles,
            "articles": arts,
        })

    # Tri final par trend_score décroissant → les sujets les plus chauds en premier
    clusters.sort(key=lambda c: -c["trend_score"])
    return clusters
