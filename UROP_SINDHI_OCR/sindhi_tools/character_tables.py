NUKTA = "\u093c"
HALANT = "\u094d"
ANUSVARA = "\u0902"
AVAGRAHA = "\u093d"

ZABAR = "\u064e"
ZER = "\u0650"
PESH = "\u064f"
SUKUN = "\u0652"
SHADDA = "\u0651"
DAGGER_ALEF = "\u0670"

ZABAR_MARKS = frozenset({ZABAR, "\u064b"})
ZER_MARKS = frozenset({ZER, "\u064d"})
PESH_MARKS = frozenset({PESH, "\u064c"})
DIACRITIC_MARKS = frozenset(
    {ZABAR, ZER, PESH, SUKUN, SHADDA, DAGGER_ALEF, "\u064b", "\u064c", "\u064d", "\u0653", "\u0654", "\u0655"}
)

ALIF_LETTERS = frozenset("اآأإٱ")
YE_LETTERS = frozenset("يىئیے")
WAW_LETTERS = frozenset("وؤۏ")
HAMZA = "ء"
AYN = "ع"
NASAL_MARK = "\u06ba"
ASPIRATION_MARK = "\u06be"

GLIDE_Y = "य"
GLIDE_W = "व"

CONSONANT_MAP = {
    "ب": "ब",
    "پ": "प",
    "ت": "त",
    "ٽ": "ट",
    "ٿ": "थ",
    "ٺ": "ठ",
    "ٻ": "ॿ",
    "ڀ": "भ",
    "ث": "स",
    "ج": "ज",
    "ڄ": "ॼ",
    "ڃ": "ञ",
    "چ": "च",
    "ڇ": "छ",
    "ڌ": "ध",
    "ڏ": "ॾ",
    "ڊ": "ड",
    "ڍ": "ढ",
    "د": "द",
    "ذ": "ज" + NUKTA,
    "ر": "र",
    "ڙ": "ड" + NUKTA,
    "ز": "ज" + NUKTA,
    "ژ": "ज" + NUKTA,
    "س": "स",
    "ش": "श",
    "ص": "स",
    "ض": "ज" + NUKTA,
    "ط": "त",
    "ظ": "ज" + NUKTA,
    "ف": "फ" + NUKTA,
    "ڦ": "फ",
    "ق": "क" + NUKTA,
    "ڪ": "क",
    "ک": "ख",
    "گ": "ग",
    "ڳ": "ॻ",
    "ڱ": "ङ",
    "ل": "ल",
    "م": "म",
    "ن": "न",
    "ڻ": "ण",
    "ه": "ह",
    "ھ": "ह",
    "خ": "ख" + NUKTA,
    "ح": "ह",
    "غ": "ग" + NUKTA,
}

PLAIN_IMPLOSIVE_MAP = {
    "ٻ": "ब",
    "ڄ": "ज",
    "ڏ": "ड",
    "ڳ": "ग",
}

ASPIRATED_MAP = {
    "ب": "भ",
    "پ": "फ",
    "ت": "थ",
    "ٽ": "ठ",
    "د": "ध",
    "ڊ": "ढ",
    "ڙ": "ढ" + NUKTA,
    "ج": "झ",
    "چ": "छ",
    "ڪ": "ख",
    "گ": "घ",
}

INDEPENDENT_VOWELS = {
    "a": "अ",
    "aa": "आ",
    "i": "इ",
    "ii": "ई",
    "u": "उ",
    "uu": "ऊ",
    "e": "ए",
    "ai": "ऐ",
    "o": "ओ",
    "au": "औ",
}

VOWEL_SIGNS = {
    "aa": "ा",
    "i": "ि",
    "ii": "ी",
    "u": "ु",
    "uu": "ू",
    "e": "े",
    "ai": "ै",
    "o": "ो",
    "au": "ौ",
}

SYMBOL_MAP = {
    "۽": "ऐं",
    "۾": "में",
}

PUNCTUATION_MAP = {
    "،": ",",
    "؛": ";",
    "؟": "?",
    "٪": "%",
    "۔": ".",
    "٫": ".",
    "٬": ",",
    "٭": "*",
    ".": ".",
    ",": ",",
    ";": ";",
    ":": ":",
    "!": "!",
    "?": "?",
    "…": "…",
}

DIGIT_MAP = {}
DEVANAGARI_DIGIT_MAP = {}
for _index in range(10):
    for _base in (0x0660, 0x06F0):
        DIGIT_MAP[chr(_base + _index)] = str(_index)
        DEVANAGARI_DIGIT_MAP[chr(_base + _index)] = chr(0x0966 + _index)

DIGIT_STYLES = ("ascii", "devanagari")
IMPLOSIVE_STYLES = ("sindhi", "plain")
HAMZA_POLICIES = ("vowel", "avagraha")
