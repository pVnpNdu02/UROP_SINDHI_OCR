import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from . import character_tables as tables
from .lexicon import Lexicon, load_default_lexicon
from .normalization import (
    DIACRITICS,
    count_variant_substitutions,
    normalize_text,
    strip_diacritics,
    tokenize,
)

UNKNOWN_POLICIES = ("keep", "drop")


def canonical_devanagari(text):
    return unicodedata.normalize("NFC", text)


@dataclass(frozen=True)
class TransliterationOptions:
    use_lexicon: bool = True
    convert_digits: bool = True
    digit_style: str = "ascii"
    nasal_as_anusvara: bool = True
    unknown_policy: str = "keep"
    implosive_style: str = "sindhi"
    hamza_policy: str = "vowel"
    merge_script_variants: bool = True

    def __post_init__(self):
        if self.unknown_policy not in UNKNOWN_POLICIES:
            raise ValueError(f"unknown_policy must be one of {UNKNOWN_POLICIES}")
        if self.digit_style not in tables.DIGIT_STYLES:
            raise ValueError(f"digit_style must be one of {tables.DIGIT_STYLES}")
        if self.implosive_style not in tables.IMPLOSIVE_STYLES:
            raise ValueError(f"implosive_style must be one of {tables.IMPLOSIVE_STYLES}")
        if self.hamza_policy not in tables.HAMZA_POLICIES:
            raise ValueError(f"hamza_policy must be one of {tables.HAMZA_POLICIES}")


@dataclass
class TransliterationReport:
    word_count: int = 0
    lexicon_hits: int = 0
    letter_count: int = 0
    unknown_characters: Counter = field(default_factory=Counter)
    omitted_characters: Counter = field(default_factory=Counter)
    variant_substitutions: Counter = field(default_factory=Counter)

    @property
    def unknown_count(self):
        return sum(self.unknown_characters.values())

    @property
    def coverage(self):
        if self.letter_count == 0:
            return 1.0
        return max(0.0, 1.0 - self.unknown_count / self.letter_count)

    def to_dict(self):
        def ordered(counter):
            return dict(sorted(counter.items(), key=lambda item: ord(item[0])))

        return {
            "word_count": self.word_count,
            "lexicon_hits": self.lexicon_hits,
            "letter_count": self.letter_count,
            "unknown_count": self.unknown_count,
            "rule_coverage": round(self.coverage, 4),
            "unknown_characters": ordered(self.unknown_characters),
            "omitted_characters": ordered(self.omitted_characters),
            "variant_substitutions": ordered(self.variant_substitutions),
        }


class Unit:
    __slots__ = ("char", "marks", "aspirated")

    def __init__(self, char):
        self.char = char
        self.marks = set()
        self.aspirated = False


def is_mark(character):
    return character in DIACRITICS or character in tables.DIACRITIC_MARKS


def parse_units(word, report=None):
    units = []
    for character in word:
        if is_mark(character):
            if units:
                units[-1].marks.add(character)
            elif report is not None:
                report.omitted_characters[character] += 1
            continue
        if (
            character == tables.ASPIRATION_MARK
            and units
            and units[-1].char in tables.ASPIRATED_MAP
            and not units[-1].aspirated
        ):
            units[-1].aspirated = True
            continue
        units.append(Unit(character))
    return units


def has_any(marks, group):
    return bool(marks & group)


def vowel_from_marks(marks):
    if has_any(marks, tables.ZER_MARKS):
        return "i"
    if has_any(marks, tables.PESH_MARKS):
        return "u"
    if has_any(marks, tables.ZABAR_MARKS):
        return "a"
    return None


def is_long_vowel_letter(unit):
    if unit is None:
        return False
    return (
        unit.char in tables.ALIF_LETTERS
        or unit.char in tables.YE_LETTERS
        or unit.char in tables.WAW_LETTERS
    )


class SindhiTransliterator:
    def __init__(self, options=None, lexicon=None):
        self.options = options or TransliterationOptions()
        if lexicon is None and self.options.use_lexicon:
            lexicon = load_default_lexicon()
        self.lexicon = lexicon if lexicon is not None else Lexicon()
        self._consonants = dict(tables.CONSONANT_MAP)
        if self.options.implosive_style == "plain":
            self._consonants.update(tables.PLAIN_IMPLOSIVE_MAP)

    def _normalize(self, text):
        return normalize_text(text, merge_variants=self.options.merge_script_variants)

    def transliterate(self, text):
        return self.transliterate_with_report(text)[0]

    def transliterate_with_report(self, text):
        report = TransliterationReport()
        if self.options.merge_script_variants:
            report.variant_substitutions.update(count_variant_substitutions(text))
        pieces = []
        digit_table = (
            tables.DEVANAGARI_DIGIT_MAP
            if self.options.digit_style == "devanagari"
            else tables.DIGIT_MAP
        )
        for kind, token in tokenize(self._normalize(text)):
            if kind == "letter":
                pieces.append(self._transliterate_word(token, report))
            elif kind == "digit":
                if self.options.convert_digits:
                    pieces.append("".join(digit_table.get(character, character) for character in token))
                else:
                    pieces.append(token)
            elif kind == "punct":
                pieces.append(tables.PUNCTUATION_MAP.get(token, token))
            else:
                pieces.append(token)
        return canonical_devanagari("".join(pieces)), report

    def transliterate_word(self, word):
        report = TransliterationReport()
        return canonical_devanagari(self._transliterate_word(self._normalize(word), report))

    def _transliterate_word(self, word, report):
        report.word_count += 1
        if self.options.use_lexicon:
            hit = self.lexicon.lookup(word)
            if hit is not None:
                report.lexicon_hits += 1
                report.letter_count += len(strip_diacritics(word))
                return hit
        units = parse_units(word, report)
        report.letter_count += len(units)
        return "".join(self._transliterate_units(units, report))

    def _independent_alif(self, unit, following, was_carrier):
        marks = unit.marks
        if was_carrier or unit.char == "آ":
            return "aa", 0
        if unit.char == "إ" or has_any(marks, tables.ZER_MARKS):
            if following is not None and following.char in tables.YE_LETTERS:
                return "ii", 1
            return "i", 0
        if has_any(marks, tables.PESH_MARKS):
            if following is not None and following.char in tables.WAW_LETTERS:
                return "uu", 1
            return "u", 0
        if following is not None and not following.marks:
            if following.char in tables.YE_LETTERS:
                return "e", 1
            if following.char in tables.WAW_LETTERS:
                return "o", 1
        return "a", 0

    def _transliterate_units(self, units, report):
        table = self._consonants
        output = []
        open_consonant = False
        carrier = False
        index = 0
        count = len(units)

        while index < count:
            unit = units[index]
            character = unit.char
            marks = unit.marks
            following = units[index + 1] if index + 1 < count else None
            at_start = index == 0
            was_carrier = carrier
            carrier = False
            skip = 0

            if character in table:
                base = tables.ASPIRATED_MAP[character] if unit.aspirated else table[character]
                output.append(base)
                if tables.SHADDA in marks:
                    output.append(tables.HALANT + base)
                open_consonant = True
                if has_any(marks, tables.ZER_MARKS):
                    if following is not None and following.char in tables.YE_LETTERS:
                        output.append(tables.VOWEL_SIGNS["ii"])
                        skip = 1
                    else:
                        output.append(tables.VOWEL_SIGNS["i"])
                    open_consonant = False
                elif has_any(marks, tables.PESH_MARKS):
                    if following is not None and following.char in tables.WAW_LETTERS:
                        output.append(tables.VOWEL_SIGNS["uu"])
                        skip = 1
                    else:
                        output.append(tables.VOWEL_SIGNS["u"])
                    open_consonant = False
                elif has_any(marks, tables.ZABAR_MARKS):
                    if following is not None and following.char in tables.ALIF_LETTERS:
                        output.append(tables.VOWEL_SIGNS["aa"])
                        skip = 1
                        open_consonant = False
                    elif following is not None and following.char in tables.YE_LETTERS:
                        output.append(tables.VOWEL_SIGNS["e"])
                        skip = 1
                        open_consonant = False
                    elif following is not None and following.char in tables.WAW_LETTERS:
                        output.append(tables.VOWEL_SIGNS["au"])
                        skip = 1
                        open_consonant = False
                elif tables.DAGGER_ALEF in marks:
                    output.append(tables.VOWEL_SIGNS["aa"])
                    open_consonant = False
                elif tables.SUKUN in marks:
                    if following is None or following.char in table:
                        output.append(tables.HALANT)
                    open_consonant = False

            elif character == tables.NASAL_MARK:
                if self.options.nasal_as_anusvara:
                    output.append(tables.ANUSVARA)
                else:
                    output.append(table["ن"])
                open_consonant = False

            elif character in tables.ALIF_LETTERS:
                if open_consonant:
                    output.append(tables.VOWEL_SIGNS["aa"])
                else:
                    key, skip = self._independent_alif(unit, following, was_carrier)
                    output.append(tables.INDEPENDENT_VOWELS[key])
                open_consonant = False

            elif character in tables.YE_LETTERS:
                if following is not None and (
                    following.char in tables.ALIF_LETTERS or following.char in tables.WAW_LETTERS
                ):
                    output.append(tables.GLIDE_Y)
                    open_consonant = True
                elif open_consonant:
                    output.append(tables.VOWEL_SIGNS["ii"])
                    open_consonant = False
                else:
                    output.append(tables.INDEPENDENT_VOWELS["ii"])
                    open_consonant = False

            elif character in tables.WAW_LETTERS:
                if open_consonant:
                    if following is not None and following.char in tables.ALIF_LETTERS:
                        output.append(tables.GLIDE_W)
                        open_consonant = True
                    else:
                        output.append(tables.VOWEL_SIGNS["o"])
                        open_consonant = False
                else:
                    output.append(tables.GLIDE_W)
                    open_consonant = True

            elif character == tables.HAMZA:
                vowel = vowel_from_marks(marks)
                if self.options.hamza_policy == "avagraha":
                    output.append(tables.AVAGRAHA)
                    open_consonant = False
                elif vowel is not None:
                    output.append(tables.INDEPENDENT_VOWELS[vowel])
                    open_consonant = False
                else:
                    report.omitted_characters[tables.HAMZA] += 1

            elif character == tables.AYN:
                vowel = vowel_from_marks(marks)
                if (
                    at_start
                    and vowel is None
                    and following is not None
                    and not following.marks
                    and is_long_vowel_letter(following)
                ):
                    if following.char in tables.ALIF_LETTERS:
                        carrier = True
                    elif following.char in tables.YE_LETTERS:
                        output.append(tables.INDEPENDENT_VOWELS["ii"])
                        skip = 1
                    else:
                        output.append(tables.INDEPENDENT_VOWELS["o"])
                        skip = 1
                    open_consonant = False
                elif vowel is not None:
                    output.append(tables.INDEPENDENT_VOWELS[vowel])
                    open_consonant = False
                elif at_start or not is_long_vowel_letter(following):
                    output.append(tables.INDEPENDENT_VOWELS["a"])
                    open_consonant = False
                else:
                    report.omitted_characters[tables.AYN] += 1

            elif character in tables.SYMBOL_MAP:
                output.append(tables.SYMBOL_MAP[character])
                open_consonant = False

            else:
                report.unknown_characters[character] += 1
                if self.options.unknown_policy == "keep":
                    output.append(character)
                open_consonant = False

            index += 1 + skip

        return output


_default_transliterator = None


def get_default_transliterator():
    global _default_transliterator
    if _default_transliterator is None:
        _default_transliterator = SindhiTransliterator()
    return _default_transliterator


def sindhi_to_devanagari(sindhi_text, options=None, lexicon=None):
    if options is None and lexicon is None:
        return get_default_transliterator().transliterate(sindhi_text)
    return SindhiTransliterator(options, lexicon).transliterate(sindhi_text)
