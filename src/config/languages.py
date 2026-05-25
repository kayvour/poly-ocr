# Maps the CLI lang code to each engine's expected format
LANG_MAP = {
    "eng": {
        "tesseract": "eng",
        "easyocr":   "en",
        "paddleocr": "en",
        "trocr":     None,  # TrOCR is English-only, lang param unused
        "donut":     None,
        "nougat":    None,
        "doctr":     None,
    },
    "hin": {
        "tesseract": "hin",
        "easyocr":   "hi",
        "paddleocr": "hi",
        "trocr":     None,
        "donut":     None,
        "nougat":    None,
        "doctr":     None,
    },
    "chi_sim": {
        "tesseract": "chi_sim",
        "easyocr":   "ch_sim",
        "paddleocr": "ch",
        "trocr":     None,
        "donut":     None,
        "nougat":    None,
        "doctr":     None,
    },
    "ara": {
        "tesseract": "ara",
        "easyocr":   "ar",
        "paddleocr": "ar",
        "trocr":     None,
        "donut":     None,
        "nougat":    None,
        "doctr":     None,
    },
}

def get_lang(lang_code: str, engine: str) -> str | None:
    """Returns the engine-specific language code, or the raw code as fallback."""
    return LANG_MAP.get(lang_code, {}).get(engine, lang_code)
    