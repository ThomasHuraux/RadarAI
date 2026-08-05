import hashlib
import requests
from datetime import datetime, timezone, timedelta

# Signal absent des autres collecteurs (RSS/arXiv/HF/S2) : la dynamique des outils/
# frameworks IA eux-mêmes (sorties de version), pas seulement la couverture presse
# ou académique qui en parle. API publique, non authentifiée (60 req/h) — largement
# suffisant pour ~15 dépôts sur un pipeline qui tourne une fois par heure.
GITHUB_API = "https://api.github.com/repos/{repo}/releases"

# Frameworks/outils IA à forte visibilité dont les sorties de version constituent
# un signal de tendance en soi (agents, inference, modèles ouverts).
TRACKED_REPOS = [
    "langchain-ai/langchain",
    "langchain-ai/langgraph",
    "microsoft/autogen",
    "crewAIInc/crewAI",
    "ggml-org/llama.cpp",
    "ollama/ollama",
    "vllm-project/vllm",
    "huggingface/transformers",
    "run-llama/llama_index",
    "openai/openai-python",
    "anthropics/anthropic-sdk-python",
    "n8n-io/n8n",
]

_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "AIRadar/1.0 (research aggregator; contact: radarai@local)",
}


def _make_id(repo: str, tag: str) -> str:
    return hashlib.md5(f"github_release:{repo}:{tag}".encode()).hexdigest()


def collect_github_releases(lookback_days: int = 2, max_per_repo: int = 3) -> list[dict]:
    """
    Collecte les releases récentes des dépôts IA suivis.

    lookback_days=2 (pas juste "aujourd'hui") : une release publiée tard dans la nuit
    UTC ne doit pas disparaître si le pipeline tourne tôt le lendemain matin.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    articles = []

    for repo in TRACKED_REPOS:
        try:
            resp = requests.get(
                GITHUB_API.format(repo=repo),
                headers=_HEADERS,
                params={"per_page": max_per_repo},
                timeout=15,
            )
            resp.raise_for_status()
            releases = resp.json()
        except Exception as e:
            print(f"[GitHub] {repo} failed: {e}")
            continue

        for rel in releases:
            # Les brouillons/pre-releases ne sont pas des annonces publiques finales.
            if rel.get("draft") or rel.get("prerelease"):
                continue

            published_at = rel.get("published_at") or rel.get("created_at")
            if not published_at:
                continue
            try:
                pub_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if pub_dt < cutoff:
                continue

            tag = rel.get("tag_name", "")
            if not tag:
                continue

            name = (rel.get("name") or tag).strip()
            # "New release — " garantit un titre suffisamment long pour passer le
            # filtre is_valid_article (>=15 caractères) même sur un tag court comme "v1.2".
            title = f"New release — {repo.split('/')[-1]} {name}"
            body = (rel.get("body") or "").strip()

            articles.append({
                "id": _make_id(repo, tag),
                "source": "github_releases",
                "title": title,
                "content": body[:2000],
                "url": rel.get("html_url", ""),
                "date": pub_dt.strftime("%Y-%m-%d"),
                "embedding": None,
                "cluster_id": -1,
            })

    return articles
