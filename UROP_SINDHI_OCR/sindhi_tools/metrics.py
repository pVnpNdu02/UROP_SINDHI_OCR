import csv
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .normalization import normalize_text


def levenshtein_distance(reference, hypothesis):
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current.append(
                min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + cost)
            )
        previous = current
    return previous[-1]


def canonical(text):
    return unicodedata.normalize("NFC", text or "").strip()


def character_error_rate(reference, hypothesis):
    reference = canonical(reference)
    hypothesis = canonical(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein_distance(reference, hypothesis) / len(reference)


def word_error_rate(reference, hypothesis):
    reference_words = canonical(reference).split()
    hypothesis_words = canonical(hypothesis).split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return levenshtein_distance(reference_words, hypothesis_words) / len(reference_words)


def alignment_operations(reference, hypothesis):
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    matrix = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        matrix[row][0] = row
    for column in range(columns):
        matrix[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            cost = 0 if reference[row - 1] == hypothesis[column - 1] else 1
            matrix[row][column] = min(
                matrix[row - 1][column] + 1,
                matrix[row][column - 1] + 1,
                matrix[row - 1][column - 1] + cost,
            )
    operations = []
    row, column = len(reference), len(hypothesis)
    while row > 0 or column > 0:
        if (
            row
            and column
            and matrix[row][column]
            == matrix[row - 1][column - 1] + (0 if reference[row - 1] == hypothesis[column - 1] else 1)
        ):
            kind = "match" if reference[row - 1] == hypothesis[column - 1] else "substitute"
            operations.append((kind, reference[row - 1], hypothesis[column - 1]))
            row -= 1
            column -= 1
        elif row and matrix[row][column] == matrix[row - 1][column] + 1:
            operations.append(("delete", reference[row - 1], ""))
            row -= 1
        else:
            operations.append(("insert", "", hypothesis[column - 1]))
            column -= 1
    operations.reverse()
    return operations


def confusion_counts(operations):
    counts = Counter()
    for kind, expected, produced in operations:
        if kind in ("substitute", "delete", "insert"):
            counts[(expected, produced)] += 1
    return counts


@dataclass
class ItemResult:
    source: str
    reference: str
    hypothesis: str
    distance: int
    character_error_rate: float
    word_error_rate: float

    @property
    def exact(self):
        return self.reference == self.hypothesis


@dataclass
class EvaluationResult:
    items: list = field(default_factory=list)

    @property
    def pair_count(self):
        return len(self.items)

    @property
    def exact_matches(self):
        return sum(item.exact for item in self.items)

    @property
    def exact_match_rate(self):
        return self.exact_matches / self.pair_count if self.pair_count else 0.0

    @property
    def mean_character_error_rate(self):
        if not self.items:
            return 0.0
        return sum(item.character_error_rate for item in self.items) / self.pair_count

    @property
    def corpus_character_error_rate(self):
        total_length = sum(len(item.reference) for item in self.items)
        if not total_length:
            return 0.0
        return sum(item.distance for item in self.items) / total_length

    @property
    def mean_word_error_rate(self):
        if not self.items:
            return 0.0
        return sum(item.word_error_rate for item in self.items) / self.pair_count

    @property
    def errors(self):
        return [item for item in self.items if not item.exact]

    def confusions(self):
        total = Counter()
        for item in self.errors:
            total.update(confusion_counts(alignment_operations(item.reference, item.hypothesis)))
        return total

    def summary(self):
        return {
            "pairs": self.pair_count,
            "exact_matches": self.exact_matches,
            "exact_match_rate": round(self.exact_match_rate, 4),
            "mean_cer": round(self.mean_character_error_rate, 4),
            "corpus_cer": round(self.corpus_character_error_rate, 4),
            "mean_wer": round(self.mean_word_error_rate, 4),
        }


def evaluate_predictions(sources, references, hypotheses):
    if not (len(sources) == len(references) == len(hypotheses)):
        raise ValueError("sources, references and hypotheses must have equal length")
    result = EvaluationResult()
    for source, reference, hypothesis in zip(sources, references, hypotheses):
        reference = canonical(reference)
        hypothesis = canonical(hypothesis)
        result.items.append(
            ItemResult(
                str(source),
                reference,
                hypothesis,
                levenshtein_distance(reference, hypothesis),
                character_error_rate(reference, hypothesis),
                word_error_rate(reference, hypothesis),
            )
        )
    return result


def load_gold_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"sindhi", "devanagari"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Gold file {path} is missing columns: {sorted(missing)}")
        for row in reader:
            sindhi = (row.get("sindhi") or "").strip()
            devanagari = (row.get("devanagari") or "").strip()
            if sindhi and devanagari:
                pairs.append((sindhi, devanagari))
    return pairs


def evaluate_transliterator(transliterator, pairs):
    sources = [source for source, _ in pairs]
    references = [reference for _, reference in pairs]
    hypotheses = [transliterator.transliterate(source) for source in sources]
    return evaluate_predictions(sources, references, hypotheses)


def normalize_image_key(value):
    key = (value or "").strip().replace("\\", "/")
    while key.startswith("./"):
        key = key[2:]
    return key


def load_ocr_references(path):
    references = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"image", "sindhi_reference"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"OCR reference file is missing columns: {sorted(missing)}")
        for row in reader:
            image = normalize_image_key(row.get("image"))
            sindhi = (row.get("sindhi_reference") or "").strip()
            devanagari = (row.get("devanagari_reference") or "").strip()
            if not image or not sindhi:
                continue
            if image in references:
                raise ValueError(f"Duplicate image in OCR reference file: {image}")
            references[image] = {"sindhi": sindhi, "devanagari": devanagari}
    return references


def write_item_csv(path, evaluation, first_column, extra=None):
    extra = extra or {}
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                first_column,
                "reference",
                "prediction",
                "edit_distance",
                "cer",
                "wer",
                "exact_match",
                "status",
            ],
        )
        writer.writeheader()
        for item in evaluation.items:
            writer.writerow(
                {
                    first_column: item.source,
                    "reference": item.reference,
                    "prediction": item.hypothesis,
                    "edit_distance": item.distance,
                    "cer": round(item.character_error_rate, 6),
                    "wer": round(item.word_error_rate, 6),
                    "exact_match": item.exact,
                    "status": extra.get(item.source, "ok"),
                }
            )


def evaluate_ocr_run(results_dir, references_path):
    results_root = Path(results_dir)
    summary_path = results_root / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"No batch summary found at {summary_path}")
    references = load_ocr_references(references_path)
    if not references:
        raise ValueError("The OCR reference file contains no usable rows")

    summary = {}
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            summary[normalize_image_key(row.get("file"))] = row

    names = []
    ocr_predictions = []
    pipeline_predictions = []
    statuses = {}
    unprocessed = []
    for image in sorted(references):
        row = summary.get(image)
        if row is None:
            unprocessed.append(image)
            continue
        names.append(image)
        payload = None
        result_name = row.get("result_json") or ""
        if row.get("status") != "error" and result_name:
            result_path = results_root / result_name
            if result_path.is_file():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload is None:
            statuses[image] = "failed"
            ocr_predictions.append("")
            pipeline_predictions.append("")
        else:
            statuses[image] = "ok"
            ocr_predictions.append(payload.get("sindhi_text", ""))
            pipeline_predictions.append(payload.get("devanagari_text", ""))

    if not names:
        raise ValueError(
            "No reference images were found in summary.csv. Compare the image column of your "
            "reference file with the file column of summary.csv."
        )

    ocr_evaluation = evaluate_predictions(
        names,
        [normalize_text(references[name]["sindhi"]) for name in names],
        ocr_predictions,
    )
    ocr_csv = results_root / "ocr_evaluation.csv"
    write_item_csv(ocr_csv, ocr_evaluation, "image", statuses)

    payload = {
        "matched_images": len(names),
        "failed_images": sorted(name for name, status in statuses.items() if status == "failed"),
        "unprocessed_reference_images": unprocessed,
        "metrics": ocr_evaluation.summary(),
        "csv_path": str(ocr_csv),
    }

    with_devanagari = [index for index, name in enumerate(names) if references[name]["devanagari"]]
    if with_devanagari:
        end_to_end = evaluate_predictions(
            [names[index] for index in with_devanagari],
            [references[names[index]]["devanagari"] for index in with_devanagari],
            [pipeline_predictions[index] for index in with_devanagari],
        )
        end_to_end_csv = results_root / "end_to_end_evaluation.csv"
        write_item_csv(end_to_end_csv, end_to_end, "image", statuses)
        payload["end_to_end_metrics"] = end_to_end.summary()
        payload["end_to_end_csv_path"] = str(end_to_end_csv)

    (results_root / "ocr_evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload
