import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LEXICON_PATH = DATA_DIR / "lexicon.json"
GOLD_PAIRS_PATH = DATA_DIR / "gold_pairs_template.csv"
OCR_REFERENCES_PATH = DATA_DIR / "ocr_references_template.csv"
DEMO_DIR = DATA_DIR / "demo"

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

OCR_LANGUAGE = "snd"
TESSERACT_ENVIRONMENT_VARIABLE = "TESSERACT_CMD"

TESSERACT_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\Public\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
)


def find_tesseract():
    configured = os.environ.get(TESSERACT_ENVIRONMENT_VARIABLE)
    if configured:
        expanded = Path(configured).expanduser()
        if expanded.is_file():
            return str(expanded)
    located = shutil.which("tesseract")
    if located:
        return located
    for candidate in TESSERACT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None
