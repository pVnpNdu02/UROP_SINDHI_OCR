import argparse
import json
import logging
import sys
from pathlib import Path

from sindhi_tools.batch import run_batch
from sindhi_tools.config import DATA_DIR, OCR_LANGUAGE, RESULTS_DIR, find_tesseract
from sindhi_tools.experiments import build_grid, run_experiment
from sindhi_tools.lexicon import load_default_lexicon
from sindhi_tools.metrics import evaluate_ocr_run, evaluate_transliterator, load_gold_pairs
from sindhi_tools.ocr_engine import OcrUnavailableError, TesseractEngine, build_engine
from sindhi_tools.pipeline import SindhiPipeline
from sindhi_tools.preprocessing import PreprocessingOptions
from sindhi_tools.transliterator import SindhiTransliterator, TransliterationOptions


def build_parser():
    parser = argparse.ArgumentParser(prog="urop_project.py", description="Sindhi OCR and Sindhi-to-Devanagari transliteration research pipeline")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--lang", default=OCR_LANGUAGE)
    check_parser.add_argument("--tesseract-cmd")
    check_parser.add_argument("--tessdata-dir")

    languages_parser = subparsers.add_parser("list-languages")
    languages_parser.add_argument("--tesseract-cmd")
    languages_parser.add_argument("--tessdata-dir")

    ocr_parser = subparsers.add_parser("ocr")
    ocr_parser.add_argument("--input-dir", required=True)
    ocr_parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    ocr_parser.add_argument("--lang", default=OCR_LANGUAGE)
    ocr_parser.add_argument("--engine", choices=("tesseract", "google", "auto"), default="tesseract")
    ocr_parser.add_argument("--psm", type=int, default=3, choices=range(0, 14))
    ocr_parser.add_argument("--tesseract-cmd")
    ocr_parser.add_argument("--tessdata-dir")
    ocr_parser.add_argument("--recursive", action="store_true")
    ocr_parser.add_argument("--no-preprocessing", action="store_true")
    ocr_parser.add_argument("--binarization", choices=("adaptive", "otsu", "none"), default="adaptive")
    ocr_parser.add_argument("--deskew", action="store_true")

    transliterate_parser = subparsers.add_parser("transliterate")
    transliterate_parser.add_argument("--text", required=True)
    transliterate_parser.add_argument("--unknown-policy", choices=("keep", "drop"), default="keep")
    transliterate_parser.add_argument("--no-lexicon", action="store_true")
    transliterate_parser.add_argument("--report", action="store_true")

    translit_eval_parser = subparsers.add_parser("evaluate-transliterator")
    translit_eval_parser.add_argument("--gold-csv", default=str(DATA_DIR / "gold_pairs_template.csv"))
    translit_eval_parser.add_argument("--output", default=str(RESULTS_DIR / "transliteration_evaluation.json"))

    ocr_eval_parser = subparsers.add_parser("evaluate-ocr")
    ocr_eval_parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    ocr_eval_parser.add_argument("--references-csv", default=str(DATA_DIR / "ocr_references_template.csv"))

    ablate_parser = subparsers.add_parser("ablate")
    ablate_parser.add_argument("--images-dir", required=True)
    ablate_parser.add_argument("--references-csv", required=True)
    ablate_parser.add_argument("--output-dir", default=str(RESULTS_DIR / "ablation"))
    ablate_parser.add_argument("--lang", default=OCR_LANGUAGE)
    ablate_parser.add_argument("--psm", type=int, nargs="+", default=[3, 6], choices=range(0, 14))
    ablate_parser.add_argument("--binarization", nargs="+", choices=("adaptive", "otsu", "none"), default=["adaptive", "otsu", "none"])
    ablate_parser.add_argument("--try-deskew", action="store_true")
    ablate_parser.add_argument("--tesseract-cmd")
    ablate_parser.add_argument("--tessdata-dir")

    return parser


def run_check(language, tesseract_cmd=None, tessdata_dir=None):
    engine = TesseractEngine(language=language, tesseract_cmd=tesseract_cmd, tessdata_dir=tessdata_dir)
    engine.check()
    version = engine.version()
    languages = ", ".join(engine.available_languages())
    print(json.dumps({"tesseract": engine.tesseract_cmd or find_tesseract(), "version": version, "requested_language": language, "installed_languages": languages, "status": "ok"}, indent=2, ensure_ascii=False))


def run_transliteration(args):
    options = TransliterationOptions(use_lexicon=not args.no_lexicon, unknown_policy=args.unknown_policy)
    transliterator = SindhiTransliterator(options=options, lexicon=load_default_lexicon() if not args.no_lexicon else None)
    output, report = transliterator.transliterate_with_report(args.text)
    if args.report:
        print(json.dumps({"input": args.text, "output": output, "report": report.to_dict()}, ensure_ascii=False, indent=2))
    else:
        print(output)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        if args.command == "check":
            run_check(args.lang, args.tesseract_cmd, args.tessdata_dir)
        elif args.command == "list-languages":
            engine = TesseractEngine(tesseract_cmd=args.tesseract_cmd, tessdata_dir=args.tessdata_dir)
            print("\n".join(engine.available_languages()))
        elif args.command == "ocr":
            engine = build_engine(args.engine, args.lang, args.tesseract_cmd, args.psm, args.tessdata_dir)
            preprocessing = PreprocessingOptions(binarization=args.binarization, deskew=args.deskew)
            pipeline = SindhiPipeline(engine, preprocessing, use_preprocessing=not args.no_preprocessing)
            def progress(position, total, row):
                print(f"[{position}/{total}] {row['status']}: {row['file']}" + (f" | {row['message']}" if row["message"] else ""))
            report = run_batch(pipeline, args.input_dir, args.output_dir, recursive=args.recursive, progress=progress)
            print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))
        elif args.command == "transliterate":
            run_transliteration(args)
        elif args.command == "evaluate-transliterator":
            pairs = load_gold_pairs(args.gold_csv)
            if not pairs:
                raise ValueError("The gold CSV contains no non-empty Sindhi/Devanagari pairs")
            result = evaluate_transliterator(SindhiTransliterator(), pairs)
            payload = result.summary()
            payload["gold_csv"] = str(args.gold_csv)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            pair_csv = output.with_name(output.stem + "_pairs.csv")
            with pair_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                import csv
                writer = csv.DictWriter(handle, fieldnames=["source", "reference", "hypothesis", "edit_distance", "cer", "wer", "exact_match"])
                writer.writeheader()
                for item in result.items:
                    writer.writerow({"source": item.source, "reference": item.reference, "hypothesis": item.hypothesis, "edit_distance": item.distance, "cer": round(item.character_error_rate, 6), "wer": round(item.word_error_rate, 6), "exact_match": item.exact})
            confusion_csv = output.with_name(output.stem + "_confusions.csv")
            with confusion_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                import csv
                writer = csv.DictWriter(handle, fieldnames=["expected_character", "predicted_character", "count"])
                writer.writeheader()
                for (expected, produced), count in result.confusions().most_common():
                    writer.writerow({"expected_character": expected, "predicted_character": produced, "count": count})
            payload["pair_results_csv"] = str(pair_csv)
            payload["character_confusions_csv"] = str(confusion_csv)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        elif args.command == "evaluate-ocr":
            payload = evaluate_ocr_run(args.results_dir, args.references_csv)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        elif args.command == "ablate":
            configs = build_grid(psm_values=tuple(args.psm), binarizations=tuple(args.binarization), try_deskew=args.try_deskew)

            def engine_factory(psm):
                return TesseractEngine(language=args.lang, tesseract_cmd=args.tesseract_cmd, page_segmentation_mode=psm, tessdata_dir=args.tessdata_dir)

            def ablation_progress(position, total, row):
                print(f"[{position}/{total}] {row['name']}: corpus_cer={row['corpus_cer']} mean_wer={row['mean_wer']} seconds/image={row['mean_seconds_per_image']}")

            payload = run_experiment(engine_factory, args.images_dir, args.references_csv, configs, args.output_dir, progress=ablation_progress)
            print(json.dumps({key: value for key, value in payload.items() if key != "ranking"}, ensure_ascii=False, indent=2))
        return 0
    except (OcrUnavailableError, FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
