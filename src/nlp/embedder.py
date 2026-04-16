import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# sentence-transformers produit de bien meilleurs embeddings (modèle neuronal),
# mais requiert PyTorch — incompatible avec Python 3.14 au moment du développement.
# On tente l'import et on bascule silencieusement sur TF-IDF+SVD si absent.
try:
    from sentence_transformers import SentenceTransformer
    _st_model = None

    def _get_st_model():
        # Chargement paresseux : le modèle (~90 Mo) n'est téléchargé qu'au premier appel
        global _st_model
        if _st_model is None:
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        return _st_model

    HAS_ST = True
except ImportError:
    HAS_ST = False

# Modèles TF-IDF+SVD conservés en mémoire entre les appels pour éviter
# de refitter à chaque article. Ils doivent être fittés sur le corpus COMPLET
# d'un même run — voir la note critique dans embed_articles().
_tfidf_model: TfidfVectorizer | None = None
_svd_model: TruncatedSVD | None = None

# Dimensionnalité de l'espace latent après SVD.
# 128 dimensions est un bon compromis : assez pour capturer la sémantique,
# assez petit pour que HDBSCAN et KMeans soient rapides.
_N_COMPONENTS = 128


def _tfidf_embed(texts: list[str]) -> np.ndarray:
    """
    Convertit une liste de textes en vecteurs denses via TF-IDF + SVD.

    TF-IDF (Term Frequency–Inverse Document Frequency) :
      - Donne un poids élevé aux mots fréquents dans UN document mais rares
        dans l'ensemble du corpus. Ex : "transformer" est discriminant,
        "the" ne l'est pas.
      - Résultat : une matrice creuse (articles × vocabulaire).

    SVD tronquée (= LSA — Latent Semantic Analysis) :
      - Réduit la matrice creuse en 128 dimensions denses.
      - Capture des relations sémantiques latentes : "LLM" et "language model"
        se retrouvent proches même s'ils ne partagent pas de mots.

    ATTENTION : le modèle doit être (re)fitté sur le corpus COMPLET à chaque
    run. Si on fitte sur un sous-ensemble et qu'on transforme le reste,
    les vecteurs n'ont pas les mêmes dimensions (ValueError inhomogeneous shape).
    """
    global _tfidf_model, _svd_model

    if _tfidf_model is None or _svd_model is None:
        # Premier appel : on fitte les deux modèles sur tout le corpus
        _tfidf_model = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
        _svd_model = TruncatedSVD(n_components=min(_N_COMPONENTS, len(texts) - 1))
        matrix = _tfidf_model.fit_transform(texts)
        embeddings = _svd_model.fit_transform(matrix)
    else:
        # Appels suivants : on réutilise le vocabulaire et l'espace SVD déjà appris
        matrix = _tfidf_model.transform(texts)
        embeddings = _svd_model.transform(matrix)

    # Normalisation L2 : chaque vecteur a une norme de 1.
    # Cela rend la distance euclidienne équivalente à la similarité cosinus,
    # ce qui est requis par HDBSCAN avec metric="euclidean".
    return normalize(embeddings).astype(np.float32)


def embed_articles(articles: list[dict]) -> list[dict]:
    """
    Calcule et attache un embedding à chaque article.
    L'embedding est stocké comme liste Python (sérialisable en JSON pour SQLite).
    """
    # On concatène titre + début du contenu pour représenter l'article
    texts = [f"{a['title']}. {a.get('content', '')[:512]}" for a in articles]

    if HAS_ST:
        model = _get_st_model()
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    else:
        print("[embedder] sentence-transformers unavailable — using TF-IDF+SVD")
        embeddings = _tfidf_embed(texts)

    for article, emb in zip(articles, embeddings):
        article["embedding"] = emb.tolist()

    return articles


def get_embeddings_matrix(articles: list[dict]) -> np.ndarray:
    """
    Reconstruit la matrice numpy des embeddings depuis les articles.
    Les embeddings sont stockés en JSON string dans SQLite — on les désérialise ici.
    """
    import json
    vecs = []
    for a in articles:
        emb = a.get("embedding")
        if isinstance(emb, str):
            emb = json.loads(emb)
        vecs.append(emb)
    return np.array(vecs, dtype=np.float32)
