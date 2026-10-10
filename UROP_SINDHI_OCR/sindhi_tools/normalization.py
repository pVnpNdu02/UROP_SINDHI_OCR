import re
import unicodedata
from collections import Counter

from .character_tables import DIACRITIC_MARKS, DIGIT_MAP, PUNCTUATION_MAP

DIACRITICS = frozenset(chr(code) for code in range(0x064B, 0x0660)) | frozenset({"\u0670"})
TATWEEL = "\u0640"
INVISIBLE_CHARACTERS = frozenset(
    {"\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\ufeff"}
)

VARIANT_PAIRS = {
    "ك": "ڪ",
    "ى": "ي",
    "ی": "ي",
    "ے": "ي",
    "ۓ": "ئ",
    "ہ": "ه",
    "ۃ": "ه",
    "ٱ": "ا",
    "ٹ": "ٽ",
    "ڈ": "ڊ",
    "ڑ": "ڙ",
}
CHARACTER_VARIANTS = str.maketrans(VARIANT_PAIRS)

HORIZONTAL_SPACE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")


def normalize_text(text, merge_variants=True):
    value = unicodedata.normalize("NFKC", text or "")
    value = "".join(
        character
        for character in value
        if character != TATWEEL and character not in INVISIBLE_CHARACTERS
    )
    if merge_variants:
        value = value.translate(CHARACTER_VARIANTS)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [HORIZONTAL_SPACE.sub(" ", line).strip() for line in value.split("\n")]
    normalized = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def count_variant_substitutions(text):
    value = unicodedata.normalize("NFKC", text or "")
    return Counter(character for character in value if character in VARIANT_PAIRS)


def strip_diacritics(text):
    return "".join(
        character
        for character in text
        if character not in DIACRITICS and character not in DIACRITIC_MARKS
    )


def is_arabic_script(character):
    code = ord(character)
    return (
        0x0600 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
        or 0x08A0 <= code <= 0x08FF
        or 0xFB50 <= code <= 0xFDFF
        or 0xFE70 <= code <= 0xFEFF
    )


def classify_character(character):
    if character in DIGIT_MAP:
        return "digit"
    if character in PUNCTUATION_MAP:
        return "punct"
    if character.isspace():
        return "space"
    if is_arabic_script(character):
        return "letter"
    return "other"


def tokenize(text):
    tokens = []
    buffer = []
    current_kind = None
    for character in text:
        kind = classify_character(character)
        if kind == "punct":
            if buffer:
                tokens.append((current_kind, "".join(buffer)))
                buffer = []
            tokens.append(("punct", character))
            current_kind = None
            continue
        if kind != current_kind and buffer:
            tokens.append((current_kind, "".join(buffer)))
            buffer = []
        current_kind = kind
        buffer.append(character)
    if buffer:
        tokens.append((current_kind, "".join(buffer)))
    return tokens


def count_script_characters(text):
    arabic = sum(
        1 for character in text if is_arabic_script(character) and character not in DIACRITICS
    )
    letters = sum(1 for character in text if not character.isspace())
    return arabic, letters


def looks_like_sindhi(text, threshold=0.5):
    arabic, letters = count_script_characters(text)
    return bool(letters) and arabic / letters >= threshold
