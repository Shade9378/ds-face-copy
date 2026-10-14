"""Safe image decoding and non-generative resizing."""

import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from preprocessing.models import FaceBounds


def load_image(source: str | Path | Image.Image, max_pixels: int) -> np.ndarray:
    """Honor EXIF orientation, composite transparency on white, and preserve RGB pixels."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            if isinstance(source, Image.Image):
                return _decode_image(source, max_pixels)
            with Image.open(source) as image:
                return _decode_image(image, max_pixels)
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as error:
        raise ValueError("Cannot decode image safely. Use a smaller PNG or JPEG.") from error


def _decode_image(image: Image.Image, max_pixels: int) -> np.ndarray:
    if image.width * image.height > max_pixels:
        raise ValueError(f"Image exceeds the configured {max_pixels:,}-pixel limit.")
    oriented = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", oriented.size, "white")
    return np.array(Image.alpha_composite(background, oriented).convert("RGB"), copy=True)


def crop_face(
    image: np.ndarray, bounds: FaceBounds, padding: float
) -> tuple[np.ndarray, FaceBounds]:
    """Pad and clamp a source bounding box; return an independent crop."""
    height, width = image.shape[:2]
    pad_x = round((bounds.right - bounds.left) * padding)
    pad_y = round((bounds.bottom - bounds.top) * padding)
    crop_bounds = FaceBounds(
        left=max(0, bounds.left - pad_x),
        top=max(0, bounds.top - pad_y),
        right=min(width, bounds.right + pad_x),
        bottom=min(height, bounds.bottom + pad_y),
    )
    if crop_bounds.right <= crop_bounds.left or crop_bounds.bottom <= crop_bounds.top:
        raise ValueError("Detected face lies outside the image.")
    return image[
        crop_bounds.top : crop_bounds.bottom, crop_bounds.left : crop_bounds.right
    ].copy(), crop_bounds


def upscale_crop(crop: np.ndarray, target: int, max_factor: float) -> tuple[np.ndarray, float]:
    """Apply bounded Lanczos interpolation; this does not restore lost facial details."""
    height, width = crop.shape[:2]
    factor = min(max_factor, max(1.0, target / min(height, width)))
    resized = Image.fromarray(crop).resize(
        (round(width * factor), round(height * factor)), Image.Resampling.LANCZOS
    )
    return np.array(resized), factor
