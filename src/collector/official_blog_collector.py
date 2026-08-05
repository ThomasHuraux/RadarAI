import hashlib
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup

# Détection par sitemap : les annonces officielles (nouveaux modèles, features, recherche)
# apparaissent souvent avec un délai dans les flux RSS génériques ou les agrégateurs
# presse — ici on lit directement la source, via <lastmod> plutôt qu'un flux RSS que
# ces deux sites ne publient pas pour leur blog.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AIRadar/1.0; +https://github.com/local/radarai)"}

# openai.com/sitemap.xml est un index de dizaines de sous-sitemaps par catégorie
# (apps-*, learn-*, webinar...) : la plupart sont des pages produit statiques, pas des
# annonces. "/release/" est le sous-sitemap qui ne contient que les vraies annonces
# produit/recherche (~79 URLs, vérifié manuellement) — pas de bruit à filtrer en plus.
_OPENAI_SITEMAP = "https://openai.com/sitemap.xml/release/"

# anthropic.com n'a qu'un seul sitemap plat (~500 URLs) mélangeant blog, carrières,
# pages légales, etc. — on filtre aux URLs sous /news/, seul espace où atterrissent
# les annonces.
_ANTHROPIC_SITEMAP = "https://www.anthropic.com/sitemap.xml"
_ANTHROPIC_NEWS_PREFIX = "https://www.anthropic.com/news/"

_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# Slugs de pages "index" (landing pages de catégorie, pas des annonces individuelles)
# qui apparaissent parfois dans le sous-sitemap "release" d'OpenAI.
_SKIP_SLUGS = {"", "index", "news", "research", "release"}


def _make_id(source: str, url: str) -> str:
    return hashlib.md5(f"{source}:{url}".encode()).hexdigest()


def _slug_title(url: str) -> str:
    """Dérive un titre lisible du dernier segment de l'URL (ex: /index/introducing-x/ -> 'Introducing X')."""
    segments = [s for s in url.rstrip("/").split("/") if s]
    slug = segments[-1] if segments else ""
    return slug.replace("-", " ").replace("_", " ").strip().title()


def _fetch_title(url: str) -> str | None:
    """Récupère le <title> réel de la page, si accessible sans rendu JS. None si échec."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception:
        return None
    soup = BeautifulSoup(resp.text, "lxml")
    if not soup.title or not soup.title.string:
        return None
    # Les <title> de ces sites suffixent le nom du site ("... \ Anthropic", "... | OpenAI")
    title = re.split(r"\s*[\\|]\s*(Anthropic|OpenAI)\s*$", soup.title.string.strip())[0].strip()
    return title or None


def _collect_from_sitemap(
    sitemap_url: str,
    source: str,
    lookback_days: int,
    max_results: int,
    url_filter=None,
    fetch_real_title: bool = False,
) -> list[dict]:
    try:
        resp = requests.get(sitemap_url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        print(f"[{source}] Sitemap fetch failed: {e}")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    entries = []
    for url_el in root.findall("sm:url", _NS):
        loc_el = url_el.find("sm:loc", _NS)
        lastmod_el = url_el.find("sm:lastmod", _NS)
        if loc_el is None or lastmod_el is None:
            continue
        loc = (loc_el.text or "").strip()
        if url_filter and not url_filter(loc):
            continue
        try:
            lastmod = datetime.fromisoformat((lastmod_el.text or "").strip().replace("Z", "+00:00"))
        except ValueError:
            continue
        if lastmod < cutoff:
            continue
        entries.append((loc, lastmod))

    # Le plus récent d'abord, borné à max_results pour éviter de scraper tout un sitemap
    # un jour où beaucoup de pages ont été mises à jour d'un coup.
    entries.sort(key=lambda e: e[1], reverse=True)
    entries = entries[:max_results]

    articles = []
    for loc, lastmod in entries:
        slug = loc.rstrip("/").split("/")[-1] if loc else ""
        if slug.lower() in _SKIP_SLUGS:
            continue

        title = None
        if fetch_real_title:
            title = _fetch_title(loc)
        if not title:
            title = _slug_title(loc)
        if not title:
            continue

        # Préfixe la source : attribution claire dans l'UI, et garantit >=15 caractères
        # (seuil de is_valid_article) même sur un titre dérivé d'un slug très court
        # (ex: OpenAI "gpt-5-6" -> "Gpt 5 6", 7 caractères).
        label = "OpenAI" if source == "openai_blog" else "Anthropic"
        title = f"{label}: {title}"

        articles.append({
            "id": _make_id(source, loc),
            "source": source,
            "title": title,
            "content": title,
            "url": loc,
            "date": lastmod.strftime("%Y-%m-%d"),
            "embedding": None,
            "cluster_id": -1,
        })

    return articles


def collect_openai_blog(lookback_days: int = 4, max_results: int = 15) -> list[dict]:
    """
    openai.com sert son HTML derrière un rendu client (Next.js) qui bloque les requêtes
    simples (page renvoyée = coquille de chargement, jamais de <title> exploitable) —
    on ne tente donc pas de récupérer le titre réel de la page, seulement celui dérivé
    du slug d'URL, qui est déjà lisible sur ce site (ex: "introducing-deep-research").
    """
    return _collect_from_sitemap(
        _OPENAI_SITEMAP, "openai_blog", lookback_days, max_results, fetch_real_title=False,
    )


def collect_anthropic_blog(lookback_days: int = 4, max_results: int = 15) -> list[dict]:
    """anthropic.com sert du HTML statique : le <title> réel de la page est récupérable."""
    return _collect_from_sitemap(
        _ANTHROPIC_SITEMAP,
        "anthropic_blog",
        lookback_days,
        max_results,
        url_filter=lambda loc: loc.startswith(_ANTHROPIC_NEWS_PREFIX) and loc != _ANTHROPIC_NEWS_PREFIX.rstrip("/"),
        fetch_real_title=True,
    )
