import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .normalization import looks_like_sindhi, normalize_text
from .preprocessing import PreprocessingOptions, load_image, preprocess_with_report, save_image
from .transliterator import SindhiTransliterator


@dataclass
class PipelineResult:
    source: str
    engine: str
    sindhi_text: str
    devanagari_text: str
    confidence: Optional[float]
    elapsed_seconds: float
    preprocessing: dict = field(default_factory=dict)
    transliteration: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "source": self.source,
            "engine": self.engine,
            "confidence": None if self.confidence is None else round(self.confidence, 2),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "sindhi_text": self.sindhi_text,
            "devanagari_text": self.devanagari_text,
            "preprocessing": self.preprocessing,
            "transliteration": self.transliteration,
            "warnings": self.warnings,
        }


class SindhiPipeline:
    def __init__(
        self,
        engine,
        preprocessing_options=None,
        transliterator=None,
        use_preprocessing=True,
        minimum_confidence=40.0,
    ):
        self.engine = engine
        self.preprocessing_options = preprocessing_options or PreprocessingOptions()
        self.transliterator = transliterator or SindhiTransliterator()
        self.use_preprocessing = use_preprocessing
        self.minimum_confidence = minimum_confidence

    def preflight(self):
        check = getattr(self.engine, "check", None)
        if check is not None:
            check()

    def prepare(self, image):
        if not self.use_preprocessing:
            return image, {}
        prepared, report = preprocess_with_report(image, self.preprocessing_options)
        return prepared, report.to_dict()

    def process_image(self, path, save_preprocessed_to=None):
        started = time.perf_counter()
        image = load_image(path)
        prepared, preprocessing_details = self.prepare(image)
        if save_preprocessed_to is not None:
            save_image(save_preprocessed_to, prepared)
        ocr = self.engine.recognize(prepared)
        sindhi_text = normalize_text(ocr.text)
        devanagari_text, report = self.transliterator.transliterate_with_report(sindhi_text)
        warnings = []
        if not sindhi_text.strip():
            warnings.append("OCR returned no text")
        elif not looks_like_sindhi(sindhi_text):
            warnings.append("Recognized text contains little Arabic-script content")
        if ocr.confidence is not None and ocr.confidence < self.minimum_confidence:
            warnings.append(f"Low OCR confidence ({ocr.confidence:.1f})")
        if report.unknown_count:
            warnings.append(f"{report.unknown_count} characters could not be transliterated")
        return PipelineResult(
            str(path),
            ocr.engine,
            sindhi_text,
            devanagari_text,
            ocr.confidence,
            time.perf_counter() - started,
            preprocessing_details,
            report.to_dict(),
            warnings,
        )

    def process_text(self, sindhi_text, source="text"):
        started = time.perf_counter()
        normalized = normalize_text(sindhi_text)
        devanagari_text, report = self.transliterator.transliterate_with_report(normalized)
        return PipelineResult(
            source,
            "none",
            normalized,
            devanagari_text,
            None,
            time.perf_counter() - started,
            {},
            report.to_dict(),
            [],
        )


def save_result(result, output_dir, stem=None):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = stem or Path(result.source).stem or "result"
    paths = {
        "sindhi": directory / f"{stem}.sindhi.txt",
        "devanagari": directory / f"{stem}.devanagari.txt",
        "json": directory / f"{stem}.json",
    }
    paths["sindhi"].write_text(result.sindhi_text + "\n", encoding="utf-8")
    paths["devanagari"].write_text(result.devanagari_text + "\n", encoding="utf-8")
    paths["json"].write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return paths
