import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .metrics import evaluate_predictions, load_ocr_references
from .normalization import normalize_text
from .ocr_engine import OcrUnavailableError
from .preprocessing import PreprocessingOptions, load_image, preprocess_image


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    preprocessing: bool
    binarization: str
    deskew: bool
    psm: int


def build_grid(
    psm_values=(3, 6),
    binarizations=("adaptive", "otsu", "none"),
    try_deskew=False,
    include_original=True,
):
    configs = []
    for psm in psm_values:
        if include_original:
            configs.append(ExperimentConfig(f"original_psm{psm}", False, "none", False, psm))
        for binarization in binarizations:
            for deskew in (False, True) if try_deskew else (False,):
                suffix = "_deskew" if deskew else ""
                configs.append(
                    ExperimentConfig(f"pre_{binarization}{suffix}_psm{psm}", True, binarization, deskew, psm)
                )
    return configs


def run_experiment(engine_factory, images_dir, references_path, configs, output_dir, progress=None):
    references = load_ocr_references(references_path)
    root = Path(images_dir)
    available = sorted(key for key in references if (root / key).is_file())
    missing = sorted(set(references) - set(available))
    if not available:
        raise ValueError(
            f"None of the {len(references)} reference images were found under {root}. "
            "Check the image column of the reference file."
        )
    if not configs:
        raise ValueError("No experiment configurations to run")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    item_rows = []

    for position, config in enumerate(configs, start=1):
        engine = engine_factory(config.psm)
        check = getattr(engine, "check", None)
        if check is not None:
            check()
        options = (
            PreprocessingOptions(binarization=config.binarization, deskew=config.deskew)
            if config.preprocessing
            else None
        )
        predictions = []
        seconds = []
        failures = 0
        for key in available:
            started = time.perf_counter()
            try:
                image = load_image(root / key)
                prepared = preprocess_image(image, options) if options is not None else image
                text = normalize_text(engine.recognize(prepared).text)
            except OcrUnavailableError:
                raise
            except Exception:
                text = ""
                failures += 1
            seconds.append(time.perf_counter() - started)
            predictions.append(text)

        evaluation = evaluate_predictions(
            available,
            [normalize_text(references[key]["sindhi"]) for key in available],
            predictions,
        )
        summary = evaluation.summary()
        row = {
            **asdict(config),
            "images": len(available),
            "failures": failures,
            "exact_match_rate": summary["exact_match_rate"],
            "mean_cer": summary["mean_cer"],
            "corpus_cer": summary["corpus_cer"],
            "mean_wer": summary["mean_wer"],
            "mean_seconds_per_image": round(sum(seconds) / len(seconds), 4),
        }
        rows.append(row)
        for item in evaluation.items:
            item_rows.append(
                {
                    "config": config.name,
                    "image": item.source,
                    "cer": round(item.character_error_rate, 6),
                    "wer": round(item.word_error_rate, 6),
                    "reference": item.reference,
                    "prediction": item.hypothesis,
                }
            )
        if progress is not None:
            progress(position, len(configs), row)

    ranked = sorted(rows, key=lambda row: (row["corpus_cer"], row["mean_seconds_per_image"]))
    csv_path = output_root / "ablation.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0].keys()))
        writer.writeheader()
        writer.writerows(ranked)
    items_path = output_root / "ablation_items.csv"
    with items_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(item_rows[0].keys()))
        writer.writeheader()
        writer.writerows(item_rows)

    payload = {
        "images_evaluated": len(available),
        "reference_images_missing": missing,
        "configurations": len(configs),
        "best_by_corpus_cer": ranked[0],
        "ranking": ranked,
        "csv_path": str(csv_path),
        "items_csv_path": str(items_path),
    }
    (output_root / "ablation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload
