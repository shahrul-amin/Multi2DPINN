from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

from data_gen.common import rectangle_from_centerline, write_geo

GridPoint = Tuple[int, int]


def generate_tree_segments(
    n_segments: int,
    segment_min_um: int,
    segment_max_um: int,
    canvas_min: int,
    canvas_max: int,
    root: GridPoint,
    rng: random.Random,
) -> List[Tuple[GridPoint, GridPoint]]:
    nodes: List[GridPoint] = [root]
    node_set = {root}
    degrees = {root: 0}
    occupied_points = {root}
    segments: List[Tuple[GridPoint, GridPoint]] = []

    while len(segments) < n_segments:
        candidates = [node for node in nodes if degrees[node] < 4]
        if not candidates:
            raise RuntimeError("No expandable nodes left for tree growth")

        rng.shuffle(candidates)
        placed = False

        for start in candidates:
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            rng.shuffle(directions)
            lengths = list(range(segment_min_um, segment_max_um + 1))
            rng.shuffle(lengths)

            for dx, dy in directions:
                for length in lengths:
                    end = (start[0] + dx * length, start[1] + dy * length)
                    if (
                        end[0] < canvas_min
                        or end[0] > canvas_max
                        or end[1] < canvas_min
                        or end[1] > canvas_max
                    ):
                        continue

                    path = [
                        (start[0] + dx * step, start[1] + dy * step)
                        for step in range(1, length + 1)
                    ]
                    if any(point in occupied_points for point in path):
                        continue

                    for point in path:
                        occupied_points.add(point)

                    if end not in node_set:
                        node_set.add(end)
                        nodes.append(end)
                        degrees[end] = 0

                    degrees[start] += 1
                    degrees[end] += 1
                    segments.append((start, end))
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break

        if not placed:
            raise RuntimeError("Unable to grow tree with current constraints")

    return segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tree-wire .geo cases")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-id", type=int, default=8000)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--segments", type=int, default=10)
    parser.add_argument("--segment-min-um", type=int, default=4)
    parser.add_argument("--segment-max-um", type=int, default=40)
    parser.add_argument("--width-um", type=float, default=1.0)
    parser.add_argument("--canvas-min", type=int, default=8)
    parser.add_argument("--canvas-max", type=int, default=248)
    parser.add_argument("--root-x", type=int, default=128)
    parser.add_argument("--root-y", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11)
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
    if args.canvas_min >= args.canvas_max:
        raise ValueError("canvas-min must be smaller than canvas-max")

    rng = random.Random(args.seed)
    root = (args.root_x, args.root_y)

    if (
        root[0] < args.canvas_min
        or root[0] > args.canvas_max
        or root[1] < args.canvas_min
        or root[1] > args.canvas_max
    ):
        raise ValueError("root must be inside the canvas")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for offset in range(args.count):
        case_id = args.start_id + offset
        geo_path = args.output_dir / f"{case_id}.geo"
        if geo_path.exists() and not args.overwrite:
            raise FileExistsError(f"{geo_path} already exists")

        segments = generate_tree_segments(
            n_segments=args.segments,
            segment_min_um=args.segment_min_um,
            segment_max_um=args.segment_max_um,
            canvas_min=args.canvas_min,
            canvas_max=args.canvas_max,
            root=root,
            rng=rng,
        )
        rectangles = [
            rectangle_from_centerline(
                (float(start[0]), float(start[1])),
                (float(end[0]), float(end[1])),
                args.width_um,
            )
            for start, end in segments
        ]
        write_geo(geo_path, rectangles)

    print(f"generated {args.count} tree geometries in {args.output_dir}")


if __name__ == "__main__":
    main()
