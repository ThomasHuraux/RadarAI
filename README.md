# AI Radar

Daily AI news — 100% open source, free, no paid LLMs.

Collects today’s articles and papers (RSS, arXiv, Semantic Scholar), groups them by topic, identifies trends, and generates a digest accessible via a web interface.

<img width="1013" height="612" alt="Capture d’écran 2026-04-17 à 00 01 07" src="https://github.com/user-attachments/assets/4fe363bf-3474-48c9-85c5-79d48d6e9189" />

---

## How It Works

```
collect → analyze → serve
```

1. **Collect** — fetches articles from configured sources
2. **Analyze** — calculates embeddings, clusters, and scores trends
3. **Serve** — displays a web dashboard with today’s hot topics

---

## Sources

| Source | Type | Articles/run |
|--------|------|-------------|
| TechCrunch, VentureBeat, MIT Tech Review, The Verge, Wired | RSS | ~50 each |
| Reddit (r/MachineLearning, r/LocalLLaMA, r/artificial) | RSS | 15–10 (capped) |
| Hacker News | Filtered RSS | ~50 |
| arXiv (cs.AI, cs.LG, cs.CL, cs.CV, cs.RO) | API | 50 |
| Semantic Scholar | API | ~100 |

---

## Technical Stack

- **Embeddings**: TF-IDF + truncated SVD (LSA, 128 dimensions) via scikit-learn — compatible with Python 3.14+. Upgrade possible with `sentence-transformers` if PyTorch is available.
- **Clustering**: HDBSCAN (density-based, no fixed k) with K-Means fallback if too much noise
- **Keywords**: TF-IDF ngrams (1-3) with AI-domain stopwords
- **Titles**: article closest to the cluster centroid (no LLM)

- **Summaries**: TextRank extraction via `sumy`
- **Persistence**: SQLite (WAL mode), complete history by date
- **Backend**: FastAPI + Jinja2
- **UI**: dark theme, responsive, pull-to-refresh (mobile + trackpad)

---

## Installation

```bash
git clone https://github.com/ThomasHuraux/RadarAI.git
cd RadarAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c “import nltk; nltk.download(‘punkt’); nltk.download(‘punkt_tab’); nltk.download(‘stopwords’)”
```

---

## Usage

```bash
# Full pipeline (collect + analyze + digest in terminal)
python main.py run

# Separate steps
python main.py collect
python main.py analyze
python main.py digest

# Web interface at http://localhost:8000
python main.py serve

# Target a specific date
python main.py collect --date 2026-04-15
python main.py analyze --date 2026-04-15
```

---

## Automation (GitHub Actions)

The `.github/workflows/daily_radar.yml` workflow runs the pipeline every day at 06:00 UTC. The SQLite database is persisted via the GitHub Actions cache between runs.

To enable it: push the repo to GitHub and enable Actions.

---

## Structure

```
RadarAI/
├── main.py                        # CLI entry point
├── requirements.txt
├── templates/
│   └── index.html                 # UI theme
├── src/
│   ├── collector/
│   │   ├── rss_collector.py       # RSS feeds (media + Reddit + HN)
│   │   ├── arxiv_collector.py     # arXiv API
│   │   └── semanticscholar_collector.py
│   ├── processor/
│   │   ├── cleaner.py             # HTML cleaning, Reddit noise filters
│   │   └── deduplicator.py        # TF-IDF cosine similarity deduplication
│   ├── nlp/
│   │   ├── embedder.py            # TF-IDF+SVD or sentence-transformers
│   │   ├── clusterer.py           # HDBSCAN + KMeans fallback
│   │   └── keywords.py            # TF-IDF n-gram keyword extraction
│   ├── trends/
│   │   └── detector.py            # Trend score, cluster centroids
│   ├── digest/
│   │   └── generator.py           # Text digest + JSON for the web
│   ├── storage/
│   │   └── db.py                  # SQLite (articles + clusters)
│   └── api/
│       └── app.py                 # FastAPI (UI + /api/refresh)
└── .github/workflows/
    └── daily_radar.yml
```

---

## License

MIT
