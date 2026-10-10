import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sindhi_tools import metrics
from sindhi_tools.batch import run_batch
from sindhi_tools.experiments import build_grid, run_experiment
from sindhi_tools.lexicon import Lexicon
from sindhi_tools.normalization import normalize_text, strip_diacritics, tokenize
from sindhi_tools.ocr_engine import OcrResult
from sindhi_tools.pipeline import SindhiPipeline
from sindhi_tools.preprocessing import PreprocessingOptions, preprocess_with_report
from sindhi_tools.transliterator import SindhiTransliterator, TransliterationOptions

NUKTA = "\u093c"


class FakeEngine:
    name = "fake"

    def __init__(self, text):
        self.text = text

    def check(self):
        return True

    def recognize(self, image):
        return OcrResult(self.text, 90.0, "fake", "snd", 0.0)


def rules_only():
    return SindhiTransliterator(options=TransliterationOptions(use_lexicon=False))


class MetricsTests(unittest.TestCase):
    def test_levenshtein_known_values(self):
        self.assertEqual(metrics.levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(metrics.levenshtein_distance("", "abc"), 3)
        self.assertEqual(metrics.levenshtein_distance(["a", "b"], ["a", "c", "b"]), 1)

    def test_cer_and_wer(self):
        self.assertEqual(metrics.character_error_rate("abcd", "abxd"), 0.25)
        self.assertEqual(metrics.character_error_rate("", ""), 0.0)
        self.assertEqual(metrics.character_error_rate("", "a"), 1.0)
        self.assertAlmostEqual(metrics.word_error_rate("a b c", "a x c"), 1 / 3)

    def test_alignment_reconstructs_both_strings(self):
        operations = metrics.alignment_operations("kitten", "sitting")
        self.assertEqual("".join(r for kind, r, _ in operations if kind != "insert"), "kitten")
        self.assertEqual("".join(h for kind, _, h in operations if kind != "delete"), "sitting")
        cost = sum(1 for kind, _, _ in operations if kind != "match")
        self.assertEqual(cost, 3)

    def test_evaluate_predictions_summary(self):
        result = metrics.evaluate_predictions(["a", "b"], ["abcd", "xy"], ["abxd", "xy"])
        summary = result.summary()
        self.assertEqual(summary["pairs"], 2)
        self.assertEqual(summary["exact_matches"], 1)
        self.assertAlmostEqual(summary["corpus_cer"], 1 / 6, places=3)
        with self.assertRaises(ValueError):
            metrics.evaluate_predictions(["a"], ["a", "b"], ["a"])


class NormalizationTests(unittest.TestCase):
    def test_variants_are_merged_but_kheh_is_kept(self):
        self.assertEqual(normalize_text("\u0643"), "\u06aa")
        self.assertEqual(normalize_text("\u06cc"), "\u064a")
        self.assertEqual(normalize_text("\u06a9"), "\u06a9")

    def test_tatweel_and_spaces(self):
        self.assertEqual(normalize_text("\u0628\u0640\u0628"), "\u0628\u0628")
        self.assertEqual(normalize_text("a  b\t c"), "a b c")

    def test_strip_diacritics(self):
        self.assertEqual(strip_diacritics("\u06aa\u0650\u062a\u0627\u0628"), "\u06aa\u062a\u0627\u0628")

    def test_tokenize_kinds(self):
        kinds = [kind for kind, _ in tokenize("\u0628 \u06f1\u06f2 \u060c")]
        self.assertEqual(kinds, ["letter", "space", "digit", "space", "punct"])


class LexiconTests(unittest.TestCase):
    def test_update_normalizes_keys(self):
        lexicon = Lexicon()
        lexicon.update({"\u0643\u062a\u0627\u0628": "X"})
        self.assertIn("\u06aa\u062a\u0627\u0628", lexicon)
        self.assertEqual(lexicon.lookup("\u06aa\u0650\u062a\u0627\u0628"), "X")

    def test_empty_key_rejected(self):
        with self.assertRaises(ValueError):
            Lexicon().add("", "x")


class TransliteratorTests(unittest.TestCase):
    def test_single_letter_rules(self):
        t = rules_only()
        self.assertEqual(t.transliterate("\u0628"), "\u092c")
        self.assertEqual(t.transliterate("\u067b"), "\u097f")
        self.assertEqual(t.transliterate("\u0680"), "\u092d")
        self.assertEqual(t.transliterate("\u06aa"), "\u0915")
        self.assertEqual(t.transliterate("\u06a9"), "\u0916")
        self.assertEqual(t.transliterate("\u062c\u06be"), "\u091d")
        self.assertEqual(t.transliterate("\u06af\u06be"), "\u0918")

    def test_long_vowels_and_final_ye(self):
        self.assertEqual(rules_only().transliterate("\u067e\u0627\u06bb\u064a"), "\u092a\u093e\u0923\u0940")

    def test_shadda_and_final_sukun(self):
        t = rules_only()
        self.assertEqual(t.transliterate("\u0628\u0651"), "\u092c\u094d\u092c")
        self.assertEqual(t.transliterate("\u0628\u0652"), "\u092c\u094d")

    def test_hamza_is_reported_when_omitted_and_ayn_is_kept(self):
        t = rules_only()
        output, report = t.transliterate_with_report("\u0621")
        self.assertEqual(output, "")
        self.assertEqual(report.omitted_characters["\u0621"], 1)
        self.assertEqual(t.transliterate("\u0639"), "\u0905")

    def test_digits_and_punctuation(self):
        self.assertEqual(rules_only().transliterate("\u06f1\u06f2"), "12")
        options = TransliterationOptions(use_lexicon=False, digit_style="devanagari")
        self.assertEqual(SindhiTransliterator(options=options).transliterate("\u06f1\u06f2"), "\u0967\u0968")
        self.assertEqual(rules_only().transliterate("\u0628\u060c \u0628"), "\u092c, \u092c")

    def test_lexicon_overrides_rules(self):
        lexicon = Lexicon({"\u06aa\u062a\u0627\u0628": "\u0915\u093f\u0924\u093e\u092c"})
        t = SindhiTransliterator(lexicon=lexicon)
        output, report = t.transliterate_with_report("\u06aa\u0650\u062a\u0627\u0628")
        self.assertEqual(output, "\u0915\u093f\u0924\u093e\u092c")
        self.assertEqual(report.lexicon_hits, 1)

    def test_unknown_policy(self):
        keep = SindhiTransliterator(options=TransliterationOptions(use_lexicon=False, unknown_policy="keep"))
        drop = SindhiTransliterator(options=TransliterationOptions(use_lexicon=False, unknown_policy="drop"))
        text = "\u0628\u0601"
        self.assertEqual(keep.transliterate(text), "\u092c\u0601")
        self.assertEqual(drop.transliterate(text), "\u092c")

    def test_invalid_options_raise(self):
        with self.assertRaises(ValueError):
            TransliterationOptions(unknown_policy="bogus")
        with self.assertRaises(ValueError):
            TransliterationOptions(digit_style="bogus")


class PreprocessingTests(unittest.TestCase):
    def test_otsu_output_is_binary_and_scaled(self):
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (40, 80), (260, 110), (0, 0, 0), -1)
        result, report = preprocess_with_report(image, PreprocessingOptions(binarization="otsu"))
        self.assertEqual(result.ndim, 2)
        self.assertTrue(set(np.unique(result)) <= {0, 255})
        self.assertEqual(report.scale, 3.0)
        self.assertEqual(report.to_dict()["binarization"], "otsu")

    def test_no_binarization_keeps_gray_levels(self):
        image = np.random.default_rng(0).integers(0, 255, (1200, 1300, 3), dtype=np.uint8)
        result, report = preprocess_with_report(image, PreprocessingOptions(binarization="none", border=0))
        self.assertEqual(report.scale, 1.0)
        self.assertGreater(len(np.unique(result)), 2)

    def test_invalid_binarization_raises(self):
        with self.assertRaises(ValueError):
            PreprocessingOptions(binarization="bogus")


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        cv2.imwrite(str(self.images / "a.png"), np.full((50, 80, 3), 255, dtype=np.uint8))
        self.text = "\u067e\u0627\u06bb\u064a"
        reference_path = self.root / "references.csv"
        with reference_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "sindhi_reference"])
            writer.writerow(["a.png", self.text])
        self.reference_path = reference_path

    def tearDown(self):
        self.tmp.cleanup()

    def test_batch_then_ocr_evaluation(self):
        pipeline = SindhiPipeline(FakeEngine(self.text), transliterator=rules_only())
        output = self.root / "results"
        report = run_batch(pipeline, self.images, output)
        self.assertEqual(report["images"], 1)
        self.assertEqual(report["errors"], 0)
        self.assertTrue((output / "summary.csv").is_file())
        payload = json.loads((output / "a.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["devanagari_text"], "\u092a\u093e\u0923\u0940")
        evaluation = metrics.evaluate_ocr_run(output, self.reference_path)
        self.assertEqual(evaluation["matched_images"], 1)
        self.assertEqual(evaluation["metrics"]["corpus_cer"], 0.0)

    def test_ablation_runs_with_stub_engine(self):
        configs = build_grid(psm_values=(3,), binarizations=("otsu",))
        self.assertEqual(len(configs), 2)
        payload = run_experiment(
            lambda psm: FakeEngine(self.text), self.images, self.reference_path, configs, self.root / "ablation"
        )
        self.assertEqual(payload["configurations"], 2)
        self.assertEqual(payload["best_by_corpus_cer"]["corpus_cer"], 0.0)


if __name__ == "__main__":
    unittest.main()
