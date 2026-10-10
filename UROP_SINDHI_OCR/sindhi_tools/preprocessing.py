from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

BINARIZATION_METHODS = ("adaptive", "otsu", "none")


@dataclass(frozen=True)
class PreprocessingOptions:
    min_side: int = 1000
    max_scale: float = 3.0
    denoise: bool = False
    binarization: str = "adaptive"
    deskew: bool = False
    deskew_max_angle: float = 10.0
    border: int = 20

    def __post_init__(self):
        if self.binarization not in BINARIZATION_METHODS:
            raise ValueError(f"binarization must be one of {BINARIZATION_METHODS}")
        if self.min_side < 0 or self.border < 0 or self.max_scale < 1.0:
            raise ValueError("min_side and border must be >= 0 and max_scale >= 1")


@dataclass
class PreprocessingReport:
    input_shape: tuple
    output_shape: tuple
    scale: float
    denoised: bool
    binarization: str
    deskew_angle: float

    def to_dict(self):
        return {
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "scale": round(self.scale, 3),
            "denoised": self.denoised,
            "binarization": self.binarization,
            "deskew_angle": round(self.deskew_angle, 2),
        }


def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def save_image(path, image):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".png"
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image for {target}")
    buffer.tofile(str(target))


def to_gray(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def scale_up(image, min_side, max_scale):
    height, width = image.shape[:2]
    shortest = min(height, width)
    if min_side <= 0 or shortest >= min_side:
        return image, 1.0
    factor = min(max_scale, min_side / float(shortest))
    resized = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    return resized, factor


def binarize(gray, method):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    if method == "otsu":
        _, result = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        block = max(15, (min(gray.shape[:2]) // 20) | 1)
        result = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 15
        )
    else:
        raise ValueError(f"Unknown binarization method: {method}")
    if (result == 255).mean() < 0.5:
        result = cv2.bitwise_not(result)
    return result


def estimate_skew(binary, max_angle=10.0, step=0.5):
    ink = cv2.bitwise_not(binary)
    height, width = ink.shape[:2]
    scale = min(1.0, 600.0 / max(height, width))
    small = cv2.resize(ink, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    center = (small.shape[1] / 2.0, small.shape[0] / 2.0)
    best_score = -1.0
    best_angle = 0.0
    for angle in np.arange(-max_angle, max_angle + step / 2.0, step):
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(
            small, matrix, (small.shape[1], small.shape[0]), flags=cv2.INTER_NEAREST
        )
        score = float(np.var(rotated.sum(axis=1).astype(np.float64)))
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle


def rotate(image, angle, fill=255):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )


def add_border(image, border, fill=255):
    if border <= 0:
        return image
    return cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=fill)


def preprocess_with_report(image, options=None):
    options = options or PreprocessingOptions()
    input_shape = tuple(image.shape)
    working, scale = scale_up(image, options.min_side, options.max_scale)
    gray = to_gray(working)
    if options.denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    result = gray
    if options.binarization != "none":
        result = binarize(gray, options.binarization)
    angle = 0.0
    if options.deskew:
        reference = result if options.binarization != "none" else binarize(gray, "otsu")
        angle = estimate_skew(reference, options.deskew_max_angle)
        if abs(angle) > 1e-6:
            result = rotate(result, angle)
    result = add_border(result, options.border)
    report = PreprocessingReport(
        input_shape=input_shape,
        output_shape=tuple(result.shape),
        scale=scale,
        denoised=options.denoise,
        binarization=options.binarization,
        deskew_angle=angle,
    )
    return result, report


def preprocess_image(image, options=None):
    return preprocess_with_report(image, options)[0]
