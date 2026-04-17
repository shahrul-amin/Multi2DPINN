from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

from data_gen.common import rectangle_from_centerline, write_geo


def build_line_rectangles(
    n_segments: int,
    segment_min_um: int,
    segment_max_um: int,
    width_um: float,
    y_center_um: float,
    x_start_um: float,
    rng: random.Random,
) -> List[tuple[float, float, float, float]]:
    rectangles: List[tuple[float, float, float, float]] = []
    x_cursor = float(x_start_um)
    y_center = float(y_center_um)
    for _ in range(n_segments):
        length_um = rng.randint(segment_min_um, segment_max_um)
        start = (x_cursor, y_center)
        end = (x_cursor + float(length_um), y_center)
        rectangles.append(rectangle_from_centerline(start, end, width_um))
        x_cursor = end[0]
    return rectangles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate straight-wire .geo cases")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--segments", type=int, default=10)
    parser.add_argument("--segment-min-um", type=int, default=4)
    parser.add_argument("--segment-max-um", type=int, default=40)
    parser.add_argument("--width-um", type=float, default=1.0)
    parser.add_argument("--y-center-um", type=float, default=128.0)
    parser.add_argument("--x-start-um", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.segments <= 0:
        raise ValueError("segments must be positive")
    if args.segment_min_um <= 0 or args.segment_max_um <= 0:
        raise ValueError("segment lengths must be positive")
    if args.segment_min_um > args.segment_max_um:
        raise ValueError("segment-min-um must be <= segment-max-um")
    if args.width_um <= 0:
        raise ValueError("width-um must be positive")

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for offset in range(args.count):
        case_id = args.start_id + offset
        geo_path = args.output_dir / f"{case_id}.geo"
        if geo_path.exists() and not args.overwrite:
            raise FileExistsError(f"{geo_path} already exists")

        rectangles = build_line_rectangles(
            n_segments=args.segments,
            segment_min_um=args.segment_min_um,
            segment_max_um=args.segment_max_um,
            width_um=args.width_um,
            y_center_um=args.y_center_um,
            x_start_um=args.x_start_um,
            rng=rng,
        )
        write_geo(geo_path, rectangles)

    print(f"generated {args.count} line geometries in {args.output_dir}")


if __name__ == "__main__":
    main()
