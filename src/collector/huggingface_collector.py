import hashlib
import requests
from datetime import datetime, timezone

# API non officielle mais stable — utilisée par le site huggingface.co/papers
_HF_PAPERS_API = "https://huggingface.co/api/daily_papers"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AIRadar/1.0; +https://github.com/local/radarai)"
}


def _make_id(paper_id: str) -> str:
    return hashlib.md5(f"huggingface:{paper_id}".encode()).hexdigest()


def collect_huggingface(max_results: int = 30) -> list[dict]:
    """
    Collecte les papers du jour depuis HuggingFace Daily Papers.

    L'API retourne les papers soumis aujourd'hui par la communauté HF,
    avec un résumé généré par IA (ai_summary) plus lisible que l'abstract arXiv brut.
    On prend ai_summary en priorité, abstract en fallback.
    """
    try:
        resp = requests.get(_HF_PAPERS_API, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        papers = resp.json()
    except Exception as e:
        print(f"[HuggingFace] Request failed: {e}")
        return []

    articles = []
    for item in papers[:max_results]:
        # L'API retourne une enveloppe avec les métadonnées de soumission au top level
        # et les données du paper dans item["paper"]
        meta = item.get("paper", {})
        paper_id = meta.get("id", "")
        title = (item.get("title") or meta.get("title") or "").strip()
        if not title:
            continue

        # ai_summary = résumé généré par HF, plus lisible que l'abstract brut
        content = (meta.get("ai_summary") or item.get("summary") or meta.get("summary") or "").strip()

        # `date` (vue/regroupement) reste "aujourd'hui" : ces papers sont dans la
        # section "Daily Papers" de HuggingFace, c'est-à-dire qu'ils ont été mis en
        # avant aujourd'hui. Leur date arXiv réelle (publishedAt = soumission initiale)
        # les ferait disparaître de la vue "aujourd'hui" si on l'utilisait pour `date` —
        # elle est conservée séparément dans `published_date`.
        view_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        real_published = (item.get("publishedAt") or meta.get("publishedAt") or "").strip()
        published_date = real_published[:10] if len(real_published) >= 10 else view_date

        articles.append({
            "id": _make_id(paper_id),
            "source": "huggingface_papers",
            "title": title,
            "content": content,
            "url": f"https://huggingface.co/papers/{paper_id}",
            "date": view_date,
            "published_date": published_date,
            "embedding": None,
            "cluster_id": -1,
        })

    return articles
