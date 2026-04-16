from functools import lru_cache

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False


@lru_cache(maxsize=512)
def _translate_cached(text: str) -> str:
    return GoogleTranslator(source="auto", target="fr").translate(text)


def translate_to_fr(text: str) -> str:
    if not text or not HAS_TRANSLATOR:
        return text
    # Skip if already looks French (heuristic: common FR words)
    fr_markers = {" le ", " la ", " les ", " un ", " une ", " des ", " est ", " sont ", " pour "}
    if any(m in f" {text.lower()} " for m in fr_markers):
        return text
    try:
        result = _translate_cached(text[:500])
        return result or text
    except Exception:
        return text


def translate_batch(texts: list[str]) -> list[str]:
    return [translate_to_fr(t) for t in texts]
