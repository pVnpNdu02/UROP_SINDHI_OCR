import json
import unicodedata
from pathlib import Path

from .config import LEXICON_PATH
from .normalization import normalize_text, strip_diacritics


class Lexicon:
    def __init__(self, entries=None):
        self._entries = {}
        if entries:
            for word, devanagari in entries.items():
                self.add(word, devanagari)

    @staticmethod
    def key_for(word):
        return strip_diacritics(normalize_text(word)).strip()

    def add(self, word, devanagari):
        key = self.key_for(word)
        if not key:
            raise ValueError("Lexicon keys cannot be empty")
        self._entries[key] = unicodedata.normalize("NFC", str(devanagari))

    def remove(self, word):
        self._entries.pop(self.key_for(word), None)

    def lookup(self, word):
        return self._entries.get(self.key_for(word))

    def update(self, other):
        for word, devanagari in other.items():
            self.add(word, devanagari)

    def items(self):
        return self._entries.items()

    def __contains__(self, word):
        return self.key_for(word) in self._entries

    def __len__(self):
        return len(self._entries)

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Lexicon file must contain a JSON object: {path}")
        return cls(payload)

    def to_json(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(dict(sorted(self._entries.items())), handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def load_default_lexicon():
    if LEXICON_PATH.is_file():
        return Lexicon.from_json(LEXICON_PATH)
    return Lexicon()
