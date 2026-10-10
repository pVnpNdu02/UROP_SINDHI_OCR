import io
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import OCR_LANGUAGE, find_tesseract

LOGGER = logging.getLogger(__name__)


class OcrUnavailableError(RuntimeError):
    pass


@dataclass
class OcrResult:
    text: str
    confidence: Optional[float]
    engine: str
    language: str
    elapsed_seconds: float


def to_pil_image(image):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(array)


def image_to_png_bytes(image):
    buffer = io.BytesIO()
    to_pil_image(image).save(buffer, format="PNG")
    return buffer.getvalue()


def assemble_text(data):
    grouped = {}
    order = []
    for index in range(len(data.get("text", []))):
        word = str(data["text"][index]).strip()
        if not word:
            continue
        key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(word)
    lines = []
    previous_paragraph = None
    for key in order:
        paragraph = key[:2]
        if previous_paragraph is not None and paragraph != previous_paragraph:
            lines.append("")
        lines.append(" ".join(grouped[key]))
        previous_paragraph = paragraph
    return "\n".join(lines)


def mean_confidence(data):
    values = []
    for raw_confidence, raw_word in zip(data.get("conf", []), data.get("text", [])):
        if not str(raw_word).strip():
            continue
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            continue
        if confidence >= 0:
            values.append(confidence)
    return sum(values) / len(values) if values else None


class TesseractEngine:
    name = "tesseract"

    def __init__(
        self,
        language=OCR_LANGUAGE,
        tesseract_cmd=None,
        page_segmentation_mode=3,
        engine_mode=1,
        extra_config="",
        tessdata_dir=None,
    ):
        self.language = language
        self.tesseract_cmd = tesseract_cmd
        self.page_segmentation_mode = page_segmentation_mode
        self.engine_mode = engine_mode
        self.extra_config = extra_config
        self.tessdata_dir = tessdata_dir
        self._pytesseract = None
        self._checked = False

    @property
    def config(self):
        parts = [f"--oem {self.engine_mode}", f"--psm {self.page_segmentation_mode}"]
        if self.extra_config:
            parts.append(self.extra_config)
        return " ".join(parts)

    def _load(self):
        if self._pytesseract is not None:
            return self._pytesseract
        try:
            import pytesseract
        except ImportError as error:
            raise OcrUnavailableError(
                "pytesseract is not installed. Run: python -m pip install -r requirements.txt"
            ) from error
        command = self.tesseract_cmd or find_tesseract()
        if command is None:
            raise OcrUnavailableError(
                "Tesseract executable was not found. Install Tesseract and set TESSERACT_CMD "
                "or pass --tesseract-cmd."
            )
        if self.tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = str(self.tessdata_dir)
        pytesseract.pytesseract.tesseract_cmd = str(command)
        try:
            pytesseract.get_tesseract_version()
        except Exception as error:
            raise OcrUnavailableError(
                f"Tesseract could not be started using {command}: {error}"
            ) from error
        self._pytesseract = pytesseract
        return pytesseract

    def version(self):
        return str(self._load().get_tesseract_version())

    def available_languages(self):
        pytesseract = self._load()
        try:
            return sorted(pytesseract.get_languages(config=""))
        except Exception as error:
            raise OcrUnavailableError(f"Could not list Tesseract languages: {error}") from error

    def check(self):
        if self._checked:
            return True
        installed = set(self.available_languages())
        wanted = [part.strip() for part in self.language.split("+") if part.strip()]
        missing = [part for part in wanted if part not in installed]
        if missing:
            raise OcrUnavailableError(
                f"Tesseract language data missing: {', '.join(missing)}. Install the corresponding "
                ".traineddata file into the tessdata directory (Linux: sudo apt install "
                f"tesseract-ocr-{missing[0]}), or point --tessdata-dir at a folder that contains it."
            )
        self._checked = True
        return True

    def recognize(self, image):
        pytesseract = self._load()
        self.check()
        started = time.perf_counter()
        try:
            data = pytesseract.image_to_data(
                to_pil_image(image),
                lang=self.language,
                config=self.config,
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractError as error:
            raise OcrUnavailableError(f"Tesseract failed: {error}") from error
        return OcrResult(
            assemble_text(data),
            mean_confidence(data),
            self.name,
            self.language,
            time.perf_counter() - started,
        )


class GoogleVisionEngine:
    name = "google"

    def __init__(self, language_hints=()):
        self.language_hints = tuple(language_hints)
        self._vision = None
        self._client = None

    def _load(self):
        if self._client is not None:
            return self._vision, self._client
        try:
            from google.cloud import vision
        except ImportError as error:
            raise OcrUnavailableError(
                "Google Vision support is optional. Install requirements-google.txt to enable it."
            ) from error
        try:
            client = vision.ImageAnnotatorClient()
        except Exception as error:
            raise OcrUnavailableError(
                f"Google Vision credentials are not configured: {error}"
            ) from error
        self._vision, self._client = vision, client
        return vision, client

    def check(self):
        self._load()
        return True

    def recognize(self, image):
        vision, client = self._load()
        started = time.perf_counter()
        try:
            payload = {"image": vision.Image(content=image_to_png_bytes(image))}
            if self.language_hints:
                payload["image_context"] = {"language_hints": list(self.language_hints)}
            response = client.document_text_detection(**payload)
        except Exception as error:
            raise OcrUnavailableError(f"Google Vision request failed: {error}") from error
        if response.error.message:
            raise OcrUnavailableError(f"Google Vision error: {response.error.message}")
        annotation = response.full_text_annotation
        confidences = [
            word.confidence
            for page in annotation.pages
            for block in page.blocks
            for paragraph in block.paragraphs
            for word in paragraph.words
        ]
        confidence = 100.0 * sum(confidences) / len(confidences) if confidences else None
        return OcrResult(
            annotation.text or "",
            confidence,
            self.name,
            "automatic-detection",
            time.perf_counter() - started,
        )


class FallbackEngine:
    name = "auto"

    def __init__(self, engines):
        if not engines:
            raise ValueError("FallbackEngine needs at least one engine")
        self.engines = list(engines)

    def check(self):
        failures = []
        for engine in self.engines:
            try:
                getattr(engine, "check", lambda: True)()
                return True
            except OcrUnavailableError as error:
                failures.append(f"{engine.name}: {error}")
        raise OcrUnavailableError("No OCR engine is available. " + " | ".join(failures))

    def recognize(self, image):
        failures = []
        for engine in self.engines:
            try:
                return engine.recognize(image)
            except OcrUnavailableError as error:
                LOGGER.warning("OCR engine %s unavailable: %s", engine.name, error)
                failures.append(f"{engine.name}: {error}")
        raise OcrUnavailableError("All OCR engines failed. " + " | ".join(failures))


ENGINE_NAMES = ("tesseract", "google", "auto")


def build_engine(
    name="tesseract",
    language=OCR_LANGUAGE,
    tesseract_cmd=None,
    page_segmentation_mode=3,
    tessdata_dir=None,
):
    def tesseract():
        return TesseractEngine(
            language=language,
            tesseract_cmd=tesseract_cmd,
            page_segmentation_mode=page_segmentation_mode,
            tessdata_dir=tessdata_dir,
        )

    if name == "tesseract":
        return tesseract()
    if name == "google":
        return GoogleVisionEngine()
    if name == "auto":
        return FallbackEngine([tesseract(), GoogleVisionEngine()])
    raise ValueError(f"Unknown OCR engine {name!r}; choose from {', '.join(ENGINE_NAMES)}")
