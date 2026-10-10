import csv
import json
import logging
import time
from pathlib import Path

from .config import SUPPORTED_EXTENSIONS
from .ocr_engine import OcrUnavailableError
from .pipeline import save_result

LOGGER = logging.getLogger(__name__)

SUMMARY_FIELDS = (
    "file",
    "status",
    "engine",
    "confidence",
    "sindhi_characters",
    "devanagari_characters",
    "words",
    "unknown_characters",
    "seconds",
    "result_json",
    "message",
)


def discover_images(directory, recursive=False):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def run_batch(pipeline, directory, output_dir, recursive=False, progress=None):
    images = discover_images(directory, recursive=recursive)
    if not images:
        raise ValueError(f"No supported image files found in {directory}")
    preflight = getattr(pipeline, "preflight", None)
    if preflight is not None:
        preflight()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    input_root = Path(directory).resolve()
    rows = []
    started = time.perf_counter()
    for position, image_path in enumerate(images, start=1):
        relative = image_path.resolve().relative_to(input_root)
        stem = "__".join(relative.with_suffix("").parts)
        row = dict.fromkeys(SUMMARY_FIELDS, "")
        row["file"] = relative.as_posix()
        try:
            result = pipeline.process_image(image_path)
            paths = save_result(result, output_root, stem=stem)
            row.update(
                status="ok" if not result.warnings else "warning",
                engine=result.engine,
                confidence="" if result.confidence is None else round(result.confidence, 2),
                sindhi_characters=len(result.sindhi_text),
                devanagari_characters=len(result.devanagari_text),
                words=result.transliteration.get("word_count", 0),
                unknown_characters=result.transliteration.get("unknown_count", 0),
                seconds=round(result.elapsed_seconds, 3),
                result_json=paths["json"].name,
                message="; ".join(result.warnings),
            )
        except OcrUnavailableError:
            raise
        except Exception as error:
            LOGGER.exception("Failed to process %s", image_path)
            row.update(status="error", message=str(error))
        rows.append(row)
        if progress is not None:
            progress(position, len(images), row)

    summary_path = output_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "images": len(images),
        "ok": sum(row["status"] == "ok" for row in rows),
        "warnings": sum(row["status"] == "warning" for row in rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "summary_csv": str(summary_path),
        "rows": rows,
    }
    (output_root / "batch_report.json").write_text(
        json.dumps(
            {key: value for key, value in report.items() if key != "rows"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report
