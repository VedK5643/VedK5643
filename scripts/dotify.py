#!/usr/bin/env python3
"""
dotify.py - turn a photo into dot-matrix / halftone art as an SVG.

Usage
-----
    python scripts/dotify.py assets/profile.jpg -o assets/portrait
    python scripts/dotify.py assets/profile.jpg -o assets/portrait --cols 100 --equalize --detail 0.5 --color
    python scripts/dotify.py assets/profile.jpg -o assets/portrait --circle --animate --color

Writes <out>.svg (color mode) or <out>-dark.svg and <out>-light.svg so the
README can swap them with <picture> + prefers-color-scheme.

Modes
-----
  dots    halftone: one circle per cell, radius scales with brightness
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
except ImportError:
    sys.exit("Pillow is required:  python -m pip install Pillow")


# --------------------------------------------------------------------------- #
# themes
# --------------------------------------------------------------------------- #

THEMES = {
    # name: (foreground, dim-foreground, background-or-None)
    "dark": ("#39d353", "#0e4429", None),
    "light": ("#216e39", "#aceebb", None),
}


# --------------------------------------------------------------------------- #
# image prep
# --------------------------------------------------------------------------- #


def square_crop(img, fx: float, fy: float):
    """Crop to 1:1 around a focus point given in 0..1 image coordinates."""
    w, h = img.size
    side = min(w, h)
    left = min(max(fx * w - side / 2, 0), w - side)
    top = min(max(fy * h - side / 2, 0), h - side)
    return img.crop((round(left), round(top), round(left) + side, round(top) + side))


def load_grid(path: Path, cols: int, contrast: float, gamma: float,
              cell_aspect: float, square: bool = False,
              focus: tuple[float, float] = (0.5, 0.5),
              equalize: bool = False, detail: float = 0.0):
    """Return (width, height, lum[y][x] in 0..1, rgb[y][x]).

    If the source has an alpha channel it is treated as a subject cutout: the
    image is flattened onto black, and the mask is carried through so nothing
    is ever drawn outside the subject.
    """
    img = ImageOps.exif_transpose(Image.open(path))

    mask = None
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        if img.split()[3].getextrema()[0] < 250:
            mask = img.split()[3]
        flat = Image.new("RGBA", img.size, (0, 0, 0, 255))
        flat.alpha_composite(img)
        img = flat
    img = img.convert("RGB")

    if square:
        img = square_crop(img, *focus)
        if mask is not None:
            mask = square_crop(mask, *focus)

    gray = img.convert("L")

    if equalize:
        binmask = mask.point(lambda v: 255 if v > 127 else 0) if mask else None
        gray = ImageOps.equalize(gray, mask=binmask)
    if detail > 0:
        radius = max(2, round(min(img.size) / 52))
        gray = gray.filter(ImageFilter.UnsharpMask(
            radius=radius, percent=round(detail * 100), threshold=0))
    if contrast != 1.0:
        gray = ImageEnhance.Contrast(gray).enhance(contrast)
        img = ImageEnhance.Contrast(img).enhance(contrast)

    w, h = img.size
    rows = max(1, round(cols * (h / w) * cell_aspect))
    small_g = gray.resize((cols, rows), Image.Resampling.LANCZOS)
    if mask is not None:
        small_m = mask.resize((cols, rows), Image.Resampling.LANCZOS)
        small_g = ImageChops.multiply(small_g, small_m)
    small_c = img.resize((cols, rows), Image.Resampling.LANCZOS)

    gp, cp = small_g.load(), small_c.load()
    rgb, lum = [], []
    for y in range(rows):
        rgb_row, lum_row = [], []
        for x in range(cols):
            rgb_row.append(cp[x, y])
            v = gp[x, y] / 255.0
            lum_row.append(min(1.0, max(0.0, v ** gamma)))
        rgb.append(rgb_row)
        lum.append(lum_row)
    return cols, rows, lum, rgb


def circle_falloff(x, y, cols, rows, feather=0.06):
    """1 inside the inscribed circle, fading to 0 just outside it."""
    nx = (x + 0.5) / cols * 2 - 1
    ny = (y + 0.5) / rows * 2 - 1
    d = math.hypot(nx, ny)
    if d <= 1 - feather:
        return 1.0
    if d >= 1 + feather:
        return 0.0
    return (1 + feather - d) / (2 * feather)


# --------------------------------------------------------------------------- #
# svg builder
# --------------------------------------------------------------------------- #


def build_svg(cols, rows, lum, rgb, *,
              fg: str, dim: str, bg: str | None,
              circle: bool, animate: bool, color: bool,
              cell: float = 10.0, pad: float = 0.8,
              min_r: float = 0.3, row_delay: float = 0.025):
    """Return an SVG string of coloured dots."""
    half = cell / 2
    vw = cols * cell + cell * 1.6
    vh = rows * cell + cell * 1.6
    margin = cell * 0.8

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {vw} {vh}" '
        f'width="{vw}" height="{vh}" '
        f'role="img" aria-label="dot-matrix portrait">'
    )

    # CSS animation classes
    if animate:
        parts.append("<style>")
        parts.append(
            "@keyframes rv{from{opacity:0}to{opacity:1}}"
            ".rw{animation:rv 0.45s ease-out both}"
        )
        for i in range(rows):
            parts.append(f".r{i}{{animation-delay:{i * row_delay:.3f}s}}")
        parts.append("</style>")

    # optional background
    if bg:
        parts.append(f'<rect width="{vw}" height="{vh}" fill="{bg}"/>')

    parts.append(f'<g transform="translate({margin},{margin})">')

    for y in range(rows):
        row_circles: list[str] = []
        for x in range(cols):
            v = lum[y][x]
            cf = circle_falloff(x, y, cols, rows) if circle else 1.0
            v *= cf
            if v < 0.02:
                continue

            r = min_r + (half * pad - min_r) * v
            cx = x * cell + half
            cy = y * cell + half

            if color:
                cr, cg, cb = rgb[y][x]
                # scale colour brightness to match the luminance-based radius
                fill = f"#{cr:02x}{cg:02x}{cb:02x}"
            else:
                # two-tone: interpolate between dim and fg
                fill = fg if v > 0.35 else dim

            row_circles.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="{fill}"/>')

        if row_circles:
            if animate:
                parts.append(f'<g class="rw r{y}">')
            else:
                parts.append("<g>")
            parts.extend(row_circles)
            parts.append("</g>")

    parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(
        description="Turn a photo into dot-matrix SVG art."
    )
    ap.add_argument("image", type=Path, help="Source image path")
    ap.add_argument("-o", "--output", type=Path, default=Path("portrait"),
                    help="Output stem (writes <stem>.svg or <stem>-dark/light.svg)")
    ap.add_argument("--cols", type=int, default=100,
                    help="Number of dot columns (default: 100)")
    ap.add_argument("--contrast", type=float, default=1.0,
                    help="Contrast adjustment (default: 1.0)")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Gamma correction (default: 1.0)")
    ap.add_argument("--circle", action="store_true",
                    help="Mask output to a circle")
    ap.add_argument("--animate", action="store_true",
                    help="Add row-by-row fade-in CSS animation")
    ap.add_argument("--color", action="store_true",
                    help="Use source pixel colours instead of two-tone")
    ap.add_argument("--square", action="store_true",
                    help="Crop input to 1:1 aspect ratio")
    ap.add_argument("--equalize", action="store_true",
                    help="Histogram-equalize for better shadow detail")
    ap.add_argument("--detail", type=float, default=0.0,
                    help="Unsharp mask strength 0..1 (default: 0)")
    ap.add_argument("--focus-x", type=float, default=0.5,
                    help="Horizontal focus point for square crop (0..1)")
    ap.add_argument("--focus-y", type=float, default=0.5,
                    help="Vertical focus point for square crop (0..1)")
    args = ap.parse_args()

    cols, rows, lum, rgb = load_grid(
        args.image, args.cols, args.contrast, args.gamma,
        cell_aspect=1.0, square=args.square,
        focus=(args.focus_x, args.focus_y),
        equalize=args.equalize, detail=args.detail,
    )

    print(f"Grid: {cols}×{rows} ({cols * rows:,} dots)")

    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.color:
        # Single colour SVG works on both dark & light themes
        svg = build_svg(cols, rows, lum, rgb,
                        fg="#ffffff", dim="#333333", bg=None,
                        circle=args.circle, animate=args.animate,
                        color=True)
        out_path = args.output.with_suffix(".svg")
        out_path.write_text(svg, encoding="utf-8")
        print(f"Wrote {out_path}  ({len(svg):,} bytes)")
    else:
        # Two separate SVGs for dark and light themes
        for theme_name, (fg, dim, bg) in THEMES.items():
            svg = build_svg(cols, rows, lum, rgb,
                            fg=fg, dim=dim, bg=bg,
                            circle=args.circle, animate=args.animate,
                            color=False)
            suffix = f"-{theme_name}.svg"
            out_path = args.output.parent / (args.output.stem + suffix)
            out_path.write_text(svg, encoding="utf-8")
            print(f"Wrote {out_path}  ({len(svg):,} bytes)")

    print("Done!")


if __name__ == "__main__":
    main()
