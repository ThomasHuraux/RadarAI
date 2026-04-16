# AI Radar

Veille IA quotidienne — 100% open source, 0€, sans LLM payant.

Collecte les articles et papers du jour (RSS, arXiv, Semantic Scholar), les regroupe par sujet, détecte les tendances, et génère un digest consultable via une interface web.

![UI screenshot](https://raw.githubusercontent.com/ThomasHuraux/RadarAI/main/.github/screenshot.png)

---

## Fonctionnement

```
collect → analyze → serve
```

1. **Collect** — récupère les articles depuis les sources configurées
2. **Analyze** — calcule les embeddings, clustérise, score les tendances
3. **Serve** — expose un dashboard web avec les hot topics du jour

---

## Sources

| Source | Type | Articles/run |
|--------|------|-------------|
| TechCrunch, VentureBeat, MIT Tech Review, The Verge, Wired | RSS | ~50 chacune |
| Reddit (r/MachineLearning, r/LocalLLaMA, r/artificial) | RSS | 15–10 (capé) |
| Hacker News | RSS filtré | ~50 |
| arXiv (cs.AI, cs.LG, cs.CL, cs.CV, cs.RO) | API | 50 |
| Semantic Scholar | API | ~100 |

---

## Stack technique

- **Embeddings** : TF-IDF + SVD tronquée (LSA, 128 dims) via scikit-learn — compatible Python 3.14+. Upgrade possible avec `sentence-transformers` si PyTorch disponible.
- **Clustering** : HDBSCAN (densité, pas de k fixe) avec fallback KMeans si trop de bruit
- **Mots-clés** : TF-IDF ngrams (1-3) avec stopwords domaine IA
- **Titres** : article le plus proche du centroïde du cluster (pas de LLM)
- **Résumés** : extractif TextRank via `sumy`
- **Persistance** : SQLite (WAL mode), historique complet par date
- **Backend** : FastAPI + Jinja2
- **UI** : dark theme, responsive, pull-to-refresh (mobile + trackpad)

---

## Installation

```bash
git clone https://github.com/ThomasHuraux/RadarAI.git
cd RadarAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

---

## Usage

```bash
# Pipeline complet (collect + analyze + digest en terminal)
python main.py run

# Étapes séparées
python main.py collect
python main.py analyze
python main.py digest

# Interface web sur http://localhost:8000
python main.py serve

# Cibler une date spécifique
python main.py collect --date 2026-04-15
python main.py analyze --date 2026-04-15
```

---

## Automatisation (GitHub Actions)

Le workflow `.github/workflows/daily_radar.yml` lance le pipeline chaque jour à 06:00 UTC. La base SQLite est conservée via le cache GitHub Actions entre les runs.

Pour l'activer : push le repo sur GitHub et activer les Actions.

---

## Structure

```
RadarAI/
├── main.py                        # CLI entry point
├── requirements.txt
├── templates/
│   └── index.html                 # UI dark theme
├── src/
│   ├── collector/
│   │   ├── rss_collector.py       # Flux RSS (media + Reddit + HN)
│   │   ├── arxiv_collector.py     # API arXiv
│   │   └── semanticscholar_collector.py
│   ├── processor/
│   │   ├── cleaner.py             # Nettoyage HTML, filtres bruit Reddit
│   │   └── deduplicator.py        # Dédup cosine similarity TF-IDF
│   ├── nlp/
│   │   ├── embedder.py            # TF-IDF+SVD ou sentence-transformers
│   │   ├── clusterer.py           # HDBSCAN + KMeans fallback
│   │   └── keywords.py            # Extraction mots-clés TF-IDF ngrams
│   ├── trends/
│   │   └── detector.py            # Trend score, titres centroïdes
│   ├── digest/
│   │   └── generator.py           # Digest texte + JSON pour le web
│   ├── storage/
│   │   └── db.py                  # SQLite (articles + clusters)
│   └── api/
│       └── app.py                 # FastAPI (UI + /api/refresh)
└── .github/workflows/
    └── daily_radar.yml
```

---

## Licence

MIT
