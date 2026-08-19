"""Save a slide thumbnail as PNG.

Usage:
    uv run scripts/save_thumbnail.py <slide_path> [--size 512] [--out thumbnail.png]
"""

import argparse
from pathlib import Path

from openslide import OpenSlide


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a slide thumbnail as PNG.")
    parser.add_argument("slide", type=Path, help="Path to the slide file (.mrxs, .svs, …)")
    parser.add_argument("--size", type=int, default=512, help="Thumbnail max dimension in pixels (default: 512)")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path (default: <slide_name>.png)")
    args = parser.parse_args()

    out = args.out or args.slide.with_suffix(".png")

    with OpenSlide(args.slide) as slide:
        thumb = slide.get_thumbnail((args.size, args.size)).convert("RGB")

    thumb.save(out)
    print(f"Saved {thumb.size[0]}×{thumb.size[1]} thumbnail → {out}")


if __name__ == "__main__":
    main()
