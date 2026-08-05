import sqlite3
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, date
from typing import Optional

# Base de données SQLite stockée dans data/radar.db à la racine du projet.
# Path(__file__) remonte depuis src/storage/ jusqu'à la racine avec .parent.parent.parent
DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "radar.db"


@contextmanager
def get_conn():
    """Context manager qui ouvre, commit/rollback ET ferme toujours la connexion."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _articles_needs_pk_migration() -> bool:
    """
    Détecte l'ancien schéma `articles` (PK sur `id` seul). Avec cette PK, recollecter
    un même article (arXiv/HF/Semantic Scholar forcent leur `date` à "aujourd'hui" à
    chaque run) faisait qu'INSERT OR REPLACE déplaçait silencieusement sa ligne vers
    la nouvelle date au lieu de le faire apparaître en plus sur les deux jours — il
    disparaissait rétroactivement de la vue d'hier alors que le cluster d'hier,
    déjà persisté dans `clusters`, référençait toujours son ancien comptage.
    """
    if not DB_PATH.exists():
        return False
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='articles'"
        ).fetchone()
    finally:
        conn.close()
    if row is None or row[0] is None:
        return False
    return "PRIMARY KEY (id, date)" not in row[0]


def _backup_db_file_before_pk_migration():
    if not DB_PATH.exists():
        return
    backup_path = DB_PATH.with_name(DB_PATH.stem + ".pre-pk-migration.bak")
    shutil.copy2(DB_PATH, backup_path)
    print(f"[db] Backed up {DB_PATH} to {backup_path} before articles PK migration.")


def init_db():
    needs_pk_migration = _articles_needs_pk_migration()
    if needs_pk_migration:
        _backup_db_file_before_pk_migration()

    with get_conn() as conn:
        # WAL (Write-Ahead Logging) : les lectures et écritures peuvent se faire
        # en parallèle sans se bloquer. Idéal pour un serveur web + cron simultanés.
        conn.executescript("""
            PRAGMA journal_mode=WAL;
        """)

        if needs_pk_migration:
            # On renomme l'ancienne table le temps de recréer la nouvelle avec la
            # PK composite ; les données sont recopiées après la CREATE TABLE plus bas.
            conn.execute("ALTER TABLE articles RENAME TO articles_old_pk_migration")

        # Migration live : si une colonne n'existe pas encore (base créée avant
        # qu'on l'ajoute), on l'ajoute sans perdre les données.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(clusters)").fetchall()}
        if cols and "name" not in cols:
            conn.execute("ALTER TABLE clusters ADD COLUMN name TEXT")
        if cols and "yesterday_count" not in cols:
            conn.execute("ALTER TABLE clusters ADD COLUMN yesterday_count INTEGER DEFAULT 0")
        if cols and "cohesion" not in cols:
            conn.execute("ALTER TABLE clusters ADD COLUMN cohesion REAL DEFAULT 0")
        if cols and "source_count" not in cols:
            conn.execute("ALTER TABLE clusters ADD COLUMN source_count INTEGER DEFAULT 0")
        if cols and "sources" not in cols:
            conn.execute("ALTER TABLE clusters ADD COLUMN sources TEXT DEFAULT '[]'")
        if cols and "low_confidence" not in cols:
            conn.execute("ALTER TABLE clusters ADD COLUMN low_confidence INTEGER DEFAULT 0")
        if cols and "labeling_method" not in cols:
            conn.execute("ALTER TABLE clusters ADD COLUMN labeling_method TEXT")
        if cols and "centroid" not in cols:
            conn.execute("ALTER TABLE clusters ADD COLUMN centroid TEXT")
        if cols and "top_entity" not in cols:
            conn.execute("ALTER TABLE clusters ADD COLUMN top_entity TEXT")
        if cols and "entity_source_count" not in cols:
            conn.execute("ALTER TABLE clusters ADD COLUMN entity_source_count INTEGER DEFAULT 0")

        article_cols = {r[1] for r in conn.execute("PRAGMA table_info(articles)").fetchall()}
        if article_cols and "cluster_fit" not in article_cols:
            conn.execute("ALTER TABLE articles ADD COLUMN cluster_fit REAL DEFAULT NULL")
        if article_cols and "duplicate_of" not in article_cols:
            conn.execute("ALTER TABLE articles ADD COLUMN duplicate_of TEXT DEFAULT NULL")
        if article_cols and "paper_id" not in article_cols:
            conn.execute("ALTER TABLE articles ADD COLUMN paper_id TEXT DEFAULT NULL")

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT NOT NULL,             -- MD5 de source:url
                source TEXT NOT NULL,         -- "arxiv", "techcrunch", "reddit_ml"...
                title TEXT NOT NULL,
                content TEXT,                 -- résumé ou corps de l'article
                url TEXT,
                date TEXT NOT NULL,           -- format YYYY-MM-DD ; date d'affichage/regroupement
                published_date TEXT,          -- format YYYY-MM-DD ; vraie date de publication si connue
                                               -- (NULL pour les lignes migrées avant l'ajout de ce champ)
                embedding TEXT,               -- vecteur JSON (liste de floats)
                cluster_id INTEGER DEFAULT -1, -- -1 = non assigné (bruit HDBSCAN)
                cluster_fit REAL DEFAULT NULL, -- similarité cosinus au centroïde du cluster
                duplicate_of TEXT DEFAULT NULL, -- id de l'article survivant si doublon
                paper_id TEXT DEFAULT NULL,    -- id arXiv normalisé (arxiv/HF/Semantic Scholar)
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id, date)         -- un même article peut réapparaître sur plusieurs jours
            );

            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER,
                date TEXT NOT NULL,
                name TEXT,                    -- titre du cluster (LLM ou fallback mots-clés)
                keywords TEXT NOT NULL,       -- JSON : liste de mots-clés TF-IDF
                trend_score REAL DEFAULT 0,
                article_count INTEGER DEFAULT 0,
                yesterday_count INTEGER DEFAULT 0,
                summary TEXT,
                top_titles TEXT,              -- JSON : liste de {title, url, source}
                cohesion REAL DEFAULT 0,      -- similarité moyenne des membres à leur centroïde
                source_count INTEGER DEFAULT 0, -- nombre de sources distinctes (dédup incluse)
                sources TEXT DEFAULT '[]',    -- JSON : liste des sources distinctes
                low_confidence INTEGER DEFAULT 0, -- 1 si cohesion < MIN_CLUSTER_FIT
                labeling_method TEXT,         -- "llm" ou "heuristic-*" — traçabilité
                centroid TEXT,                -- JSON : centroïde d'embedding du cluster (matching cross-jour)
                top_entity TEXT,               -- entité IA connue (modèle/produit) la plus corroborée
                entity_source_count INTEGER DEFAULT 0, -- nb de familles de sources citant top_entity
                PRIMARY KEY (id, date)        -- un cluster peut exister sur plusieurs jours
            );

            CREATE TABLE IF NOT EXISTS label_cache (
                articles_hash TEXT PRIMARY KEY,
                name TEXT,
                summary TEXT,
                labeling_method TEXT,
                coherence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            -- Index sur date pour les requêtes "articles du jour" (très fréquentes)
            CREATE INDEX IF NOT EXISTS idx_articles_date ON articles(date);
            -- Index sur cluster_id pour reconstituer les articles d'un cluster
            CREATE INDEX IF NOT EXISTS idx_articles_cluster ON articles(cluster_id);
        """)

        if needs_pk_migration:
            old_cols = {r[1] for r in conn.execute("PRAGMA table_info(articles_old_pk_migration)").fetchall()}
            new_cols = [r[1] for r in conn.execute("PRAGMA table_info(articles)").fetchall()]
            # Ne copie que les colonnes qui existaient déjà dans l'ancien schéma ;
            # les colonnes nouvelles (ex: published_date) restent NULL/défaut pour
            # les lignes migrées, faute de savoir rétroactivement leur vraie valeur.
            common = [c for c in new_cols if c in old_cols]
            cols_sql = ", ".join(common)
            conn.execute(
                f"INSERT INTO articles ({cols_sql}) SELECT {cols_sql} FROM articles_old_pk_migration"
            )
            migrated = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            conn.execute("DROP TABLE articles_old_pk_migration")
            print(f"[db] Migrated {migrated} articles to composite PK (id, date).")


def upsert_article(article: dict):
    # INSERT OR REPLACE sur la PK (id, date) : si le même article est recollecté un
    # autre jour (arXiv/HF/Semantic Scholar forcent leur `date` à "aujourd'hui" à
    # chaque run), il apparaît EN PLUS sur ce nouveau jour plutôt que de s'y déplacer
    # en écrasant sa ligne d'hier — cf. _articles_needs_pk_migration ci-dessus.
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO articles
                (id, source, title, content, url, date, published_date, embedding, cluster_id,
                 cluster_fit, duplicate_of, paper_id)
            VALUES
                (:id, :source, :title, :content, :url, :date, :published_date,
                 :embedding, :cluster_id, :cluster_fit, :duplicate_of, :paper_id)
        """, {
            "id": article["id"],
            "source": article["source"],
            "title": article["title"],
            "content": article.get("content", ""),
            "url": article.get("url", ""),
            "date": article["date"],
            # Vraie date de publication quand la source la fournit ; sinon égale à
            # `date` (cas des sources qui n'ont jamais menti dessus : RSS, GitHub,
            # blogs officiels). NULL seulement pour les lignes migrées avant ce champ.
            "published_date": article.get("published_date", article["date"]),
            # L'embedding est stocké comme JSON string car SQLite n'a pas de type ARRAY
            "embedding": json.dumps(article["embedding"]) if article.get("embedding") else None,
            "cluster_id": article.get("cluster_id", -1),
            "cluster_fit": article.get("cluster_fit"),
            "duplicate_of": article.get("duplicate_of"),
            "paper_id": article.get("paper_id"),
        })


def get_articles_by_date(target_date: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM articles WHERE date = ?", (target_date,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_clusterable_articles_by_date(target_date: str) -> list[dict]:
    # Exclut les articles marqués comme doublons d'un autre (cf. deduplicator.py) —
    # ils restent visibles dans l'explorateur d'articles mais ne doivent pas
    # participer à l'embedding/clustering, sous peine de gonfler artificiellement
    # le article_count d'un cluster avec plusieurs fois la même actualité.
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM articles WHERE date = ? AND duplicate_of IS NULL", (target_date,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_duplicate_sources_by_date(target_date: str) -> dict[str, list[str]]:
    # {survivor_id: [sources des articles fusionnés vers lui]} — permet de recréditer
    # la diversité de sources d'un cluster même quand les doublons ont été exclus
    # du clustering (cf. build_clusters).
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT duplicate_of, source FROM articles WHERE date = ? AND duplicate_of IS NOT NULL",
            (target_date,),
        ).fetchall()
    result: dict[str, list[str]] = {}
    for r in rows:
        result.setdefault(r["duplicate_of"], []).append(r["source"])
    return result


def update_article_cluster(article_id: str, target_date: str, cluster_id: int, cluster_fit: float | None = None):
    # Mise à jour ciblée après clustering — plus efficace qu'un upsert complet.
    # `target_date` est requis depuis le passage à la PK composite (id, date) : sans
    # lui, un article recollecté sur plusieurs jours verrait TOUTES ses lignes
    # (toutes dates confondues) écrasées par le cluster_id du run du jour.
    with get_conn() as conn:
        conn.execute(
            "UPDATE articles SET cluster_id = ?, cluster_fit = ? WHERE id = ? AND date = ?",
            (cluster_id, cluster_fit, article_id, target_date),
        )


def save_clusters(clusters: list[dict], target_date: str):
    # On supprime et recrée les clusters du jour à chaque run.
    # Plus simple que de gérer des UPDATE/INSERT différenciés.
    with get_conn() as conn:
        conn.execute("DELETE FROM clusters WHERE date = ?", (target_date,))
        for c in clusters:
            conn.execute("""
                INSERT INTO clusters
                    (id, date, name, keywords, trend_score, article_count, yesterday_count,
                     summary, top_titles, cohesion, source_count, sources, low_confidence,
                     labeling_method, centroid, top_entity, entity_source_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                c["id"],
                target_date,
                c.get("name", ""),
                json.dumps(c["keywords"]),
                c.get("trend_score", 0),
                c.get("article_count", 0),
                c.get("yesterday_count", 0),
                c.get("summary", ""),
                json.dumps(c.get("top_titles", [])),
                c.get("cohesion", 0),
                c.get("source_count", 0),
                json.dumps(c.get("sources", [])),
                int(c.get("low_confidence", False)),
                c.get("labeling_method"),
                json.dumps(c["centroid"]) if c.get("centroid") is not None else None,
                c.get("top_entity"),
                c.get("entity_source_count", 0),
            ))


def get_clusters_by_date(target_date: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM clusters WHERE date = ? ORDER BY trend_score DESC",
            (target_date,),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        # On désérialise les champs JSON au retour
        d["keywords"] = json.loads(d["keywords"])
        d["top_titles"] = json.loads(d["top_titles"])
        d["sources"] = json.loads(d["sources"]) if d.get("sources") else []
        d["low_confidence"] = bool(d.get("low_confidence", 0))
        d["centroid"] = json.loads(d["centroid"]) if d.get("centroid") else None
        result.append(d)
    return result


def get_cached_label(articles_hash: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT name, summary, labeling_method, coherence FROM label_cache WHERE articles_hash = ?",
            (articles_hash,),
        ).fetchone()
    return dict(row) if row else None


def save_cached_label(articles_hash: str, name: str, summary: str, labeling_method: str, coherence: float):
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO label_cache (articles_hash, name, summary, labeling_method, coherence)
            VALUES (?, ?, ?, ?, ?)
        """, (articles_hash, name, summary, labeling_method, coherence))


def count_articles_by_date(target_date: str) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as n FROM articles WHERE date = ?", (target_date,)
        ).fetchone()
    return row["n"]


def count_articles_by_source(target_date: str) -> dict:
    # Retourne un dict {source: count} pour afficher la répartition dans la stats bar
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT source, COUNT(*) as n FROM articles WHERE date = ? GROUP BY source",
            (target_date,),
        ).fetchall()
    return {r["source"]: r["n"] for r in rows}


def article_exists(article_id: str, target_date: str) -> bool:
    # Vérifie id + date : permet au même article de réapparaître d'un jour sur l'autre
    # (ex: papier arXiv collecté hier qui doit aussi figurer dans le digest d'aujourd'hui)
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM articles WHERE id = ? AND date = ?", (article_id, target_date)
        ).fetchone()
    return row is not None
