import re

# Entités IA connues (labs, familles de modèles, produits) à repérer explicitement dans
# le texte. But : détecter "plusieurs sources indépendantes parlent du MÊME modèle/
# produit précis", un signal plus fort et plus direct que la similarité sémantique du
# clustering — deux articles peuvent atterrir dans des clusters embeddings différents
# tout en parlant tous les deux de "GPT-5.6", ou au contraire un cluster peut être
# thématiquement cohérent (presse + réseaux) sans qu'aucun article ne cite un même nom.
#
# Liste à maintenir à mesure que sortent de nouveaux modèles/produits — même logique
# que TRACKED_REPOS dans github_collector.py ou les stopwords IA dans keywords.py :
# une énumération figée plutôt qu'une détection générique, volontairement, pour rester
# 100% local et gratuit (pas de NER via LLM cloud, pas de modèle spaCy supplémentaire).
_ENTITY_PATTERNS = [
    re.compile(r"\bgpt-?\d+(?:\.\d+)*[a-z]?\b", re.IGNORECASE),
    re.compile(r"\bchatgpt\b", re.IGNORECASE),
    re.compile(r"\bo\d(?:-(?:mini|pro))?\b", re.IGNORECASE),
    re.compile(r"\bclaude(?:\s+(?:opus|sonnet|haiku|fable|mythos))?(?:\s+\d+(?:\.\d+)?)?\b", re.IGNORECASE),
    re.compile(r"\bclaude code\b", re.IGNORECASE),
    re.compile(r"\bgemini(?:\s+\d+(?:\.\d+)?)?(?:\s+(?:pro|ultra|flash|nano))?\b", re.IGNORECASE),
    re.compile(r"\bgemma\s*\d*(?:\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\bllama\s*\d+(?:\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\bgrok[- ]?\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:mistral|mixtral|codestral)(?:\s+\w+)?\b", re.IGNORECASE),
    re.compile(r"\bdeepseek[- ]?[\w.]*\b", re.IGNORECASE),
    re.compile(r"\bqwen\s*\d*(?:\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\bsora\b", re.IGNORECASE),
    re.compile(r"\bmidjourney\b", re.IGNORECASE),
    re.compile(r"\bstable diffusion\b", re.IGNORECASE),
    re.compile(r"\brunway(?:ml)?\b", re.IGNORECASE),
    re.compile(r"\belevenlabs\b", re.IGNORECASE),
    re.compile(r"\bperplexity\b", re.IGNORECASE),
    re.compile(r"\bdevin\b", re.IGNORECASE),
    re.compile(r"\bcursor\b", re.IGNORECASE),
    re.compile(r"\bwindsurf\b", re.IGNORECASE),
    re.compile(r"\bgithub copilot\b", re.IGNORECASE),
    re.compile(r"\bcopilot\b", re.IGNORECASE),
    re.compile(r"\bn8n\b", re.IGNORECASE),
    re.compile(r"\bvllm\b", re.IGNORECASE),
    re.compile(r"\bollama\b", re.IGNORECASE),
    re.compile(r"\blangchain\b", re.IGNORECASE),
    re.compile(r"\blanggraph\b", re.IGNORECASE),
    re.compile(r"\bautogen\b", re.IGNORECASE),
    re.compile(r"\bcrewai\b", re.IGNORECASE),
]


def extract_entities(text: str) -> set[str]:
    """Renvoie l'ensemble des entités IA connues mentionnées dans le texte, normalisées
    (minuscules, espaces réduits) pour servir de clé de regroupement stable."""
    if not text:
        return set()
    found = set()
    for pattern in _ENTITY_PATTERNS:
        for m in pattern.finditer(text):
            normalized = re.sub(r"\s+", " ", m.group(0).strip().lower())
            found.add(normalized)
    return found
