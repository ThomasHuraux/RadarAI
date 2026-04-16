from sklearn.feature_extraction.text import TfidfVectorizer
import re


# Mots à exclure du calcul des mots-clés.
# sklearn fournit une liste "english" standard, mais elle ne couvre pas
# les termes génériques du domaine IA ni le bruit des interfaces web (Reddit).
_AI_STOPWORDS = {
    "said", "says", "also", "new", "one", "like", "just", "use",
    "using", "used", "year", "years", "way", "make", "making",
    "made", "get", "got", "work", "working", "need", "needs",
    "even", "still", "first", "last", "week", "day", "time",
    "researchers", "research", "company", "companies", "team",
    "according", "report", "model", "models",
    # Bruit Reddit : ces mots apparaissent dans les métadonnées des posts,
    # pas dans le contenu — ils pollueraient les mots-clés de clusters Reddit.
    "comments", "comment", "link", "submitted", "score", "points",
    "posted", "upvotes", "reddit", "subreddit", "thread",
    # Termes génériques sans valeur discriminante
    "article", "read", "post", "posts", "people", "thing", "things",
    "help", "ceo", "user", "users", "data", "many", "much",
    "know", "think", "want", "really", "actually", "would", "could",
    "different", "better", "good", "bad", "big", "small", "high",
    "well", "may", "will", "can", "let", "take", "give", "right",
}


def extract_keywords(texts: list[str], top_n: int = 8) -> list[str]:
    """
    Extrait les mots-clés les plus représentatifs d'un ensemble de textes.

    Méthode : TF-IDF sur le corpus du cluster, puis sélection des n-grammes
    avec le score global le plus élevé (somme des TF-IDF sur tous les articles).

    Pourquoi des n-grammes jusqu'à 3 (ngram_range=(1,3)) ?
    "large language model" est plus informatif que "language" seul.
    Les trigrammes capturent des expressions techniques caractéristiques.
    """
    if not texts:
        return []

    # Nettoyage minimal : on garde lettres, chiffres et espaces
    clean = [re.sub(r"[^a-zA-Z0-9 ]", " ", t).lower() for t in texts]

    try:
        vec = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=2000,
            stop_words="english",  # stopwords sklearn en première passe
            min_df=1,              # on accepte les termes qui n'apparaissent qu'une fois
        )
        matrix = vec.fit_transform(clean)

        # Score global de chaque terme = somme de ses scores TF-IDF sur tous les articles
        scores = matrix.sum(axis=0).A1
        vocab = vec.get_feature_names_out()

        # Tri décroissant par score
        ranked = sorted(zip(vocab, scores), key=lambda x: -x[1])

        keywords = []
        for term, _ in ranked:
            words = term.split()
            # Rejeter si un des mots est dans notre liste de stopwords domaine
            if any(w in _AI_STOPWORDS for w in words):
                continue
            # Rejeter les tokens de 1-2 caractères (abréviations non informatives)
            if any(len(w) <= 2 for w in words):
                continue
            keywords.append(term)
            if len(keywords) >= top_n:
                break

        return keywords
    except Exception:
        return []


def get_cluster_name(keywords: list[str]) -> str:
    # Fallback quand le titre centroïde n'est pas disponible :
    # on forge un nom lisible à partir des 3 meilleurs mots-clés
    if not keywords:
        return "Unknown topic"
    top = keywords[:3]
    return " · ".join(w.title() for w in top)
