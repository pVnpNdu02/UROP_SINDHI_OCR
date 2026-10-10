# Sindhi OCR and Sindhi-to-Devanagari Transliteration

This repository is a research prototype for reading Sindhi text from images, normalizing OCR output, applying a configurable rule-based transliteration, and measuring results against manually verified references.

## Status

Research prototype that extends the original UROP script (kept unchanged in `legacy/urop_prototype_original.py`). It implements Tesseract OCR with an optional Google Vision engine, image preprocessing, rule-based transliteration with a word lexicon, and evaluation tooling. It does not include trained or adaptive machine-learning models, NLP components or a dataset, and it reports no accuracy figures: those require manually verified references.

The rule-based transliteration tables are a starting point, not a linguistically validated standard. The output must be reviewed by a fluent Sindhi reader before it is used as a gold standard or claimed as accurate. CER and WER are meaningful only when the reference text has been manually verified and test examples are kept separate from examples used to tune the lexicon.

## Project layout

```text
UROP_SINDHI_OCR/
├── urop_project.py
├── requirements.txt
├── requirements-google.txt
├── README.md
├── sindhi_tools/
│   ├── __init__.py
│   ├── config.py
│   ├── character_tables.py
│   ├── normalization.py
│   ├── lexicon.py
│   ├── metrics.py
│   ├── preprocessing.py
│   ├── ocr_engine.py
│   ├── transliterator.py
│   ├── pipeline.py
│   └── batch.py
├── data/
│   ├── lexicon.json
│   ├── gold_pairs_template.csv
│   └── ocr_references_template.csv
├── tests/
│   └── test_core.py
└── results/
```

## 1. Install Python

Use Python 3.10 or newer. Confirm your version:

```bash
python --version
```

On some Linux or macOS installations, use `python3` instead of `python`.

## 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Windows Command Prompt:

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install Python dependencies

Run from the repository root:

```bash
python -m pip install -r requirements.txt
```

These packages provide the Python-side OCR wrapper, image loading and preprocessing. Tesseract itself is a separate system program and must also be installed.

## 4. Install Tesseract OCR

Install Tesseract for your operating system. Then ensure the Sindhi language model `snd.traineddata` is available in Tesseract's `tessdata` directory. Tesseract may be installed without the Sindhi model, so installing the executable alone may not be sufficient.

Official references:

- Tesseract installation: https://tesseract-ocr.github.io/tessdoc/Installation.html
- Tesseract language data: https://github.com/tesseract-ocr/tessdata

Confirm the executable is accessible:

```bash
tesseract --version
```

On Windows, if `tesseract` is not on PATH, set the executable path for the current PowerShell session:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

For Command Prompt:

```bat
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

For macOS or Linux, set `TESSERACT_CMD` only if the executable is not on PATH:

```bash
export TESSERACT_CMD=/usr/bin/tesseract
```

If the language model is in a non-standard folder, set `TESSDATA_PREFIX` to the directory Tesseract expects for its language data, as documented by your installation.

## 5. Check the installation

Run:

```bash
python urop_project.py check --lang snd
```

The command checks the executable and confirms the requested language model is installed. If it reports that `snd` is missing, install the corresponding model before running OCR.

List available languages:

```bash
python urop_project.py list-languages
```

## 6. Add image files

Create a directory such as `data/images/` and put your source scans or photographs in it. Supported extensions are PNG, JPG/JPEG, TIF/TIFF, BMP and WEBP.

Example:

```text
data/
├── images/
│   ├── page_001.png
│   ├── page_002.jpg
│   └── book_1/
│       └── page_003.tif
├── lexicon.json
├── gold_pairs_template.csv
└── ocr_references_template.csv
```

For recursive processing, nested image folders are included only when you pass `--recursive`.

## 7. Run OCR and transliteration

For all images directly inside `data/images`:

```bash
python urop_project.py ocr --input-dir data/images --output-dir results --lang snd
```

To include nested directories:

```bash
python urop_project.py ocr --input-dir data/images --output-dir results --lang snd --recursive
```

To compare a different binarization approach:

```bash
python urop_project.py ocr --input-dir data/images --output-dir results_otsu --lang snd --recursive --binarization otsu
```

To disable preprocessing and send the original image to the OCR engine:

```bash
python urop_project.py ocr --input-dir data/images --output-dir results_original --lang snd --recursive --no-preprocessing
```

To enable the optional skew correction step:

```bash
python urop_project.py ocr --input-dir data/images --output-dir results_deskew --lang snd --recursive --deskew
```

The application writes `.sindhi.txt`, `.devanagari.txt` and `.json` files for each image, along with `summary.csv` and `batch_report.json`. Read the warnings column in `summary.csv`; a low OCR confidence is not a guarantee of incorrect text, and a high confidence is not a guarantee of correctness.

## 8. Transliterate text without OCR

```bash
python urop_project.py transliterate --text "سنڌي متن" --report
```

To bypass the word lexicon:

```bash
python urop_project.py transliterate --text "سنڌي متن" --no-lexicon --report
```

To drop characters that are not covered by the current rules:

```bash
python urop_project.py transliterate --text "سنڌي متن" --unknown-policy drop --report
```

Keeping unknown characters is the default because silently deleting input can conceal unsupported-script coverage problems. The reported `rule_coverage` indicates the fraction of processed characters not flagged as unknown; it is not an accuracy score.

## 9. Improve word mappings through the lexicon

Edit `data/lexicon.json` with manually reviewed word-level mappings. Use valid JSON and Unicode text, for example:

```json
{
  "سنڌ": "सिंध"
}
```

The example is a formatting demonstration and should be verified by a fluent reader before being treated as a correct gold mapping. Lexicon entries take precedence over character-level rules. Do not use evaluation/test examples to tune the lexicon if you intend to report held-out performance.

## 10. Evaluate transliteration

Copy `data/gold_pairs_template.csv` to `data/gold_pairs.csv` and enter manually reviewed pairs using exactly these columns:

```csv
sindhi,devanagari
سنڌ,सिंध
```

The displayed pair is an example of the required format; verify the transcription before treating it as a reference. Run:

```bash
python urop_project.py evaluate-transliterator --gold-csv data/gold_pairs.csv --output results/transliteration_evaluation.json
```

The report includes exact-match rate, mean CER, corpus CER and mean WER. Separate `_pairs.csv` and `_confusions.csv` files contain per-pair scores and character-level error counts. CER is computed over Unicode code points after NFC normalization; WER uses whitespace-separated tokens. The evaluation is only as reliable as the reference pairs. Results from the prototype rules should not be presented as validated transliteration quality without expert review.

## 11. Evaluate OCR separately

Copy `data/ocr_references_template.csv` to `data/ocr_references.csv`. The columns must be:

- `image`: image path relative to the input directory used for OCR, using forward slashes
- `sindhi_reference`: the manually verified Sindhi transcription of that image

Example format:

```csv
image,sindhi_reference
page_001.png,سنڌي متن
book_1/page_003.tif,سنڌي متن
```

Then run OCR on the image directory and evaluate the resulting OCR text:

```bash
python urop_project.py ocr --input-dir data/images --output-dir results --lang snd --recursive
python urop_project.py evaluate-ocr --results-dir results --references-csv data/ocr_references.csv
```

Reference transcriptions are passed through the same Unicode normalization as the OCR output (compatibility forms, tatweel, and merging of look-alike letters such as Arabic kaf and Urdu yeh) before scoring, so typing-convention differences are not counted as OCR errors. Sindhi kheh (U+06A9) and kaf (U+06AA) are kept distinct.

Evaluation outputs are `results/ocr_evaluation.csv` and `results/ocr_evaluation.json`. If paths do not match, compare `image` values in the reference file with the `file` column in `results/summary.csv`.

### Preprocessing and page-segmentation ablation

To compare preprocessing settings and Tesseract page-segmentation modes on the same manually referenced images:

```bash
python urop_project.py ablate --images-dir data/images --references-csv data/ocr_references.csv --output-dir results/ablation --lang snd
```

Add `--try-deskew` to include skew correction, and `--psm 3 6 7` or `--binarization otsu none` to change the grid. Results are written to `ablation.csv`, `ablation_items.csv` and `ablation.json`. Rankings from a small or unrepresentative image set should not be generalized.

## 12. Run tests

From the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests use a stub OCR engine, so they do not require Tesseract.

These tests cover implementation behavior, not the linguistic quality of the mapping table. Passing software tests does not establish OCR accuracy or transliteration validity.

## 13. Optional Google Cloud Vision engine

Google Vision is optional and may require a Google Cloud project, billing, API enablement and application credentials. Install its additional dependency with:

```bash
python -m pip install -r requirements-google.txt
```

Configure credentials using the process described in Google's official documentation. The code does not force a Sindhi language hint; recognition capability for a specific Sindhi document must be verified with real samples. `auto` is an availability fallback: it tries Tesseract first and Google Vision only if Tesseract is unavailable or raises the project's OCR-unavailable error. It does not choose the most accurate output.

## Research reporting checklist

- Keep a held-out test set separate from lexicon development.
- Preserve source images and manually checked reference transcriptions.
- Report dataset size, image conditions, OCR engine/version, language model, preprocessing settings and page-segmentation mode.
- Report CER and WER on OCR text, transliteration output and end-to-end output separately.
- Include failures and blank outputs; do not silently remove difficult images.
- Compare preprocessing settings on the same test images.
- Report runtime as measured, not estimated.
- Have a fluent Sindhi reader review the mapping rules and gold references before drawing linguistic conclusions.
