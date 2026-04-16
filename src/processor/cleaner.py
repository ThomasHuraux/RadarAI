import re
from bs4 import BeautifulSoup


def clean_html(text: str) -> str:
    # Les flux RSS contiennent souvent du HTML brut (<p>, <a>, &amp;, etc.)
    # BeautifulSoup extrait le texte lisible en ignorant les balises.
    if not text:
        return ""
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator=" ", strip=True)


def normalize(text: str) -> str:
    # Étape 1 : supprimer les balises HTML résiduelles
    text = clean_html(text)
    # Étape 2 : supprimer les URLs (http://... ou https://...) — pas de valeur pour le clustering
    text = re.sub(r"http\S+", "", text)
    # Étape 3 : normaliser les espaces multiples en un seul espace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Reddit publie des pseudo-articles dont le titre est juste l'interface UI :
# "link", "submitted", "comments"... Ces entrées n'ont aucun contenu utile.
_REDDIT_NOISE = {"link", "submitted", "comments", "score", "by", "posted"}


def is_valid_article(article: dict) -> bool:
    title = article.get("title", "")
    words = set(title.lower().split())
    # Si tous les mots du titre sont des mots-bruit Reddit, on rejette
    if words and words.issubset(_REDDIT_NOISE):
        return False
    # Un titre de moins de 10 caractères n'est pas un vrai titre
    if len(title) < 10:
        return False
    return True


def clean_article(article: dict) -> dict:
    # Nettoie titre et contenu sur place avant d'écrire en base
    article["title"] = normalize(article.get("title", ""))
    article["content"] = normalize(article.get("content", ""))
    return article
