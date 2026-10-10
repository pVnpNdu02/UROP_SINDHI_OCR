from .lexicon import Lexicon, load_default_lexicon
from .metrics import (
    character_error_rate,
    evaluate_transliterator,
    levenshtein_distance,
    load_gold_pairs,
    word_error_rate,
)
from .normalization import normalize_text, strip_diacritics, tokenize
from .transliterator import (
    SindhiTransliterator,
    TransliterationOptions,
    TransliterationReport,
    sindhi_to_devanagari,
)

__version__ = "2.0.0"

__all__ = [
    "Lexicon",
    "SindhiTransliterator",
    "TransliterationOptions",
    "TransliterationReport",
    "character_error_rate",
    "evaluate_transliterator",
    "levenshtein_distance",
    "load_default_lexicon",
    "load_gold_pairs",
    "normalize_text",
    "sindhi_to_devanagari",
    "strip_diacritics",
    "tokenize",
    "word_error_rate",
]
