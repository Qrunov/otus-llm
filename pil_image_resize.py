"""Подготовка PIL-картинок для ViT/VLM: крошечные (например 1×1) дают тензор [3,1,1] и предупреждение transformers."""

from __future__ import annotations

from PIL import Image


def ensure_min_side(img: Image.Image, min_side: int) -> Image.Image:
    if min_side <= 0:
        return img.convert("RGB")
    img = img.convert("RGB")
    w, h = img.size
    if w < 1 or h < 1:
        raise ValueError("invalid image size")
    if min(w, h) >= min_side:
        return img
    scale = min_side / float(min(w, h))
    nw = max(min_side, int(round(w * scale)))
    nh = max(min_side, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def cap_max_edge(img: Image.Image, max_edge: int | None) -> Image.Image:
    """Уменьшает очень большие кадры (меньше RAM до encode; ViT всё равно режет вход)."""
    if not max_edge or max_edge <= 0:
        return img
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    out = img.copy()
    out.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return out
