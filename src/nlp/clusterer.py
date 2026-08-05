import numpy as np

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Plancher/plafond de sécurité pour le seuil de cohésion dynamique ci-dessous : quel
# que soit le backend d'embeddings, on ne laisse jamais le seuil dériver hors de cette
# bande (sinon un jour dégénéré pourrait soit tout gater en bruit, soit ne rien gater).
_DYNAMIC_FIT_FLOOR = 0.35
_DYNAMIC_FIT_CEILING = 0.75

# Valeur de repli utilisée quand le corpus du jour est trop petit pour estimer une
# distribution fiable (cf. _dynamic_cohesion_threshold). Calibrée à l'origine sur des
# embeddings nomic-embed-text réels (deux journées de production, `main.py inspect`
# sur 2026-07-15/16) : un plancher sous le bas des clusters sains observés (~0.74-0.88).
MIN_CLUSTER_FIT = 0.55

# Taille minimale d'un cluster après rétrogradation des membres mal ajustés — un
# "sujet" porté par un seul article restant n'en est plus un.
MIN_CLUSTER_SIZE_AFTER_GATING = 2

# En dessous de ce nombre de similarités observées, la distribution du jour est trop
# petite pour être un signal fiable — on retombe sur le plancher statique MIN_CLUSTER_FIT.
_MIN_SAMPLES_FOR_DYNAMIC_THRESHOLD = 20


def _dynamic_cohesion_threshold(embs: np.ndarray, labels: np.ndarray) -> float:
    """
    Calcule le seuil de cohésion du jour à partir de la distribution réelle des
    similarités membre→centroïde, au lieu d'une constante figée une fois pour toutes.

    MIN_CLUSTER_FIT=0.55 avait été calibré à la main sur deux journées de production
    avec un backend d'embeddings donné (cf. commentaire ci-dessus) — tout changement de
    backend (modèle Ollama différent, autre fournisseur d'embeddings) exigeait de le
    recalibrer manuellement. Ici, on prend le 10e percentile des similarités
    membre→centroïde observées CE jour-là : ça rejette systématiquement la queue la
    moins cohérente de chaque cluster, quelle que soit la bande de valeurs (resserrée et
    haute pour nomic-embed-text, plus large pour un autre modèle), sans intervention
    manuelle. Bridé entre _DYNAMIC_FIT_FLOOR et _DYNAMIC_FIT_CEILING pour éviter qu'un
    jour dégénéré (ex: corpus quasi identique, ou au contraire très dispersé) ne
    désactive le gating ou ne gate tout en bruit.
    """
    unique_labels = {l for l in labels.tolist() if l != -1}
    sims: list[float] = []
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if len(idx) < 2:
            continue
        centroid = normalize(embs[idx].mean(axis=0, keepdims=True))[0]
        sims.extend((embs[idx] @ centroid).tolist())

    if len(sims) < _MIN_SAMPLES_FOR_DYNAMIC_THRESHOLD:
        return MIN_CLUSTER_FIT

    threshold = float(np.percentile(sims, 10))
    return min(_DYNAMIC_FIT_CEILING, max(_DYNAMIC_FIT_FLOOR, threshold))


def _gate_by_cohesion(embs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Rétrograde en bruit (-1) tout article trop éloigné du centroïde de son cluster,
    puis dissout les clusters devenus trop petits une fois ces membres retirés.

    Retourne aussi le seuil effectivement utilisé, pour que les appelants (ex: le
    calcul de `low_confidence` par cluster) restent cohérents avec ce qui a réellement
    filtré les articles ce jour-là plutôt que de comparer à une constante différente.
    """
    labels = labels.copy()
    unique_labels = {l for l in labels.tolist() if l != -1}
    threshold = _dynamic_cohesion_threshold(embs, labels)

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        centroid = normalize(embs[idx].mean(axis=0, keepdims=True))[0]
        sims = embs[idx] @ centroid
        for i, sim in zip(idx, sims):
            if sim < threshold:
                labels[i] = -1

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if 0 < len(idx) < MIN_CLUSTER_SIZE_AFTER_GATING:
            labels[idx] = -1

    return labels, threshold


def _attach_cluster_fit(articles: list[dict], embs: np.ndarray, labels: np.ndarray) -> None:
    """Attache à chaque article sa similarité cosinus au centroïde final (None si bruit)."""
    unique_labels = {l for l in labels.tolist() if l != -1}
    centroids = {
        label: normalize(embs[np.where(labels == label)[0]].mean(axis=0, keepdims=True))[0]
        for label in unique_labels
    }

    for article, emb, label in zip(articles, embs, labels):
        article["cluster_fit"] = float(emb @ centroids[label]) if label in centroids else None


def cluster_articles(articles: list[dict], embeddings: np.ndarray) -> tuple[list[dict], float]:
    """
    Groupe les articles par sujet via clustering dans l'espace d'embeddings.

    Stratégie : HDBSCAN en priorité, KMeans en fallback, puis un filtre de cohésion
    commun aux deux (voir _gate_by_cohesion) qui rejette les membres mal ajustés
    au lieu de les forcer dans un groupe auquel ils ne ressemblent pas.

    HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) :
      - Ne nécessite pas de spécifier le nombre de clusters à l'avance.
      - Détecte des clusters de forme arbitraire (pas forcément sphériques).
      - Marque les articles trop isolés comme "bruit" (label = -1).
      - Inconvénient : sur un espace peu discriminant (TF-IDF creux, ou même des
        embeddings neuronaux denses où "actualité IA" domine la variance), il peut
        soit tout classifier en bruit, soit à l'inverse fusionner plusieurs sujets
        distincts en un seul gros cluster stable. D'où le fallback KMeans pour le
        premier cas, et `cluster_selection_method="leaf"` pour le second (voir plus
        bas).

    KMeans :
      - Requiert un k fixe, mais garantit que chaque article est assigné.
      - Moins fin que HDBSCAN mais robuste sur les espaces TF-IDF.
      - Sans le filtre de cohésion, un article isolé thématiquement serait quand
        même assigné à son centroïde le plus proche, aussi éloigné soit-il.
    """
    # Pas la peine de clusterer 4 articles — un seul groupe suffit
    if len(articles) < 5:
        for a in articles:
            a["cluster_id"] = 0
            a["cluster_fit"] = None
        return articles, MIN_CLUSTER_FIT

    # Renormaliser pour garantir des vecteurs unitaires
    # (HDBSCAN euclidean ≡ cosine similarity sur vecteurs normalisés)
    embs = normalize(embeddings)

    if HAS_HDBSCAN and len(articles) >= 10:
        clusterer = hdbscan.HDBSCAN(
            # Calibré sur des embeddings nomic-embed-text réels (cf. commentaire de
            # MIN_CLUSTER_FIT ci-dessus) : un plancher fixe de 4 (avec une légère
            # croissance pour de plus gros corpus) produit des clusters étroits et
            # thématiquement homogènes ; l'ancienne formule N/15 (~7-8 pour un corpus
            # quotidien type) était trop permissive et laissait `cluster_selection_method`
            # par défaut ("eom") fusionner des sujets distincts en un seul gros cluster.
            min_cluster_size=max(4, len(articles) // 30),
            # min_samples=2 (au lieu de 1, le réglage le plus permissif de HDBSCAN) :
            # limite l'effet de chaînage single-link qui laissait un cluster s'étendre
            # de proche en proche jusqu'à absorber des articles sans rapport entre eux.
            min_samples=2,
            metric="euclidean",
            # epsilon=0.0 : pas de fusion forcée entre clusters distincts.
            # Valeur 0.1 fusionnait des sujets trop éloignés → clusters incohérents.
            cluster_selection_epsilon=0.0,
            # "leaf" plutôt que le défaut "eom" (excess of mass) : eom préfère le
            # cluster le plus "stable" dans la hiérarchie, ce qui sur des embeddings
            # nomic-embed-text tend à choisir un unique gros cluster parent regroupant
            # plusieurs sous-sujets (ex. observé : un cluster de 42 articles mélangeant
            # jeux Roblox, sécurité d'agents IA et modèles de conduite autonome).
            # "leaf" sélectionne les clusters les plus fins de l'arbre, ce qui donne des
            # groupes plus petits mais réellement homogènes (vérifié sur données réelles
            # 2026-07-15/16 — cf. `main.py inspect`).
            cluster_selection_method="leaf",
        )
        labels = clusterer.fit_predict(embs)

        # Avec cluster_selection_method="leaf", un taux de bruit élevé (60-70%) est
        # normal et attendu (la plupart des articles d'une journée n'ont pas 4+
        # équivalents thématiques ce jour-là) — ce n'est plus un signe d'échec de
        # HDBSCAN. On ne bascule sur KMeans que si la clusterisation est vraiment
        # dégénérée (quasi aucune structure trouvée), pas juste sélective.
        valid = sum(1 for l in labels if l >= 0)
        if valid < len(articles) * 0.15:
            n_clusters = min(12, max(4, len(articles) // 6))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embs)
    else:
        # Moins de 10 articles ou HDBSCAN absent → KMeans directement
        n_clusters = min(12, max(4, len(articles) // 6))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embs)

    labels, threshold = _gate_by_cohesion(embs, labels)
    _attach_cluster_fit(articles, embs, labels)

    for article, label in zip(articles, labels):
        article["cluster_id"] = int(label)

    return articles, threshold
