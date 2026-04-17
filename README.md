# AI Radar

Daily AI news — 100% open source, free, no paid LLMs.

Collects today’s articles and papers (RSS, arXiv, Semantic Scholar), groups them by topic, identifies trends, and generates a digest accessible via a web interface.

<img width="1013" height="612" alt="Screenshot 2026-04-17 at 00:01:07" src="https://github.com/user-attachments/assets/3d71d28e-9df3-431c -826f-21e866784f01" />


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
- **Abstracts**: e

Translated with DeepL.com (free version)
