from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def deduplicate(articles: list[dict], threshold: float = 0.85) -> list[dict]:
    """
    Supprime les doublons sémantiques par similarité cosinus TF-IDF.

    Pourquoi TF-IDF ici et pas les embeddings SVD ?
    La déduplication se fait AVANT l'embedding — elle doit être rapide et
    ne pas dépendre d'un modèle global fitté. TF-IDF simple suffit pour
    détecter deux articles qui reprennent les mêmes mots.

    Pourquoi cosine_similarity et pas distance euclidienne ?
    La similarité cosinus est invariante à la longueur du texte :
    un article de 50 mots et un de 500 mots sur le même sujet auront
    un cosinus proche de 1, mais une distance euclidienne très grande.

    threshold=0.85 : en dessous, les articles sont considérés différents.
    Au-dessus, le second est supprimé (on garde le premier rencontré).
    """
    if len(articles) <= 1:
        return articles

    # On concatène titre + début du contenu pour la comparaison
    texts = [f"{a['title']} {a['content'][:300]}" for a in articles]

    try:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        # sim[i][j] = score de similarité entre l'article i et l'article j (entre 0 et 1)
        sim = cosine_similarity(matrix)
    except Exception:
        return articles

    keep = []
    dropped = set()

    for i in range(len(articles)):
        if i in dropped:
            continue
        keep.append(articles[i])
        # Marquer tous les articles trop similaires à i comme "à ignorer"
        for j in range(i + 1, len(articles)):
            if sim[i, j] >= threshold:
                dropped.add(j)

    return keep
