from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Tuple

Point = Tuple[float, float]
Rectangle = Tuple[float, float, float, float]

_NUMBER_PATTERN = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")


def parse_rectangles(geo_path: Path) -> List[Rectangle]:
    rectangles: List[Rectangle] = []
    with geo_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("Rectangle"):
                continue
            values = _NUMBER_PATTERN.findall(line)
            if len(values) < 6:
                raise ValueError(f"Invalid Rectangle record in {geo_path}: {line}")
            x = float(values[1])
            y = float(values[2])
            dx = float(values[4])
            dy = float(values[5])
            rectangles.append((x, y, dx, dy))
    if not rectangles:
        raise ValueError(f"No Rectangle records found in {geo_path}")
    return rectangles


def segment_centerline_from_rectangle(rect: Rectangle) -> Tuple[Point, Point, int]:
    x, y, dx, dy = rect
    node_0 = (x, y)
    node_1 = (x + dx, y + dy)

    if abs(dx) > abs(dy):
        node_0_2nd = (node_0[0], node_0[1] + dy)
        node_1_2nd = (node_0[0] + dx, node_0[1])
        node_0 = ((node_0[0] + node_0_2nd[0]) / 2.0, (node_0[1] + node_0_2nd[1]) / 2.0)
        node_1 = ((node_1[0] + node_1_2nd[0]) / 2.0, (node_1[1] + node_1_2nd[1]) / 2.0)
        if dx > 0:
            return node_0, node_1, 0
        return node_1, node_0, 0

    node_0_2nd = (node_0[0] + dx, node_0[1])
    node_1_2nd = (node_0[0], node_0[1] + dy)
    node_0 = ((node_0[0] + node_0_2nd[0]) / 2.0, (node_0[1] + node_0_2nd[1]) / 2.0)
    node_1 = ((node_1[0] + node_1_2nd[0]) / 2.0, (node_1[1] + node_1_2nd[1]) / 2.0)
    if dy > 0:
        return node_0, node_1, 1
    return node_1, node_0, 1


def rectangle_from_centerline(start: Point, end: Point, width: float) -> Rectangle:
    if width <= 0.0:
        raise ValueError("width must be positive")
    if start[1] == end[1]:
        dx = end[0] - start[0]
        if dx == 0.0:
            raise ValueError("segment length is zero")
        x = start[0]
        y = start[1] - width / 2.0
        dy = width
        return x, y, dx, dy
    if start[0] == end[0]:
        dy = end[1] - start[1]
        if dy == 0.0:
            raise ValueError("segment length is zero")
        x = start[0] - width / 2.0
        y = start[1]
        dx = width
        return x, y, dx, dy
    raise ValueError("centerline segment must be axis-aligned")


def write_geo(geo_path: Path, rectangles: Sequence[Rectangle]) -> None:
    lines = ['SetFactory("OpenCASCADE");\n']
    for idx, (x, y, dx, dy) in enumerate(rectangles, start=1):
        lines.append(
            f"Rectangle({idx}) = {{{x:.6f}, {y:.6f}, 0, {dx:.6f}, {dy:.6f}, 0}};\n"
        )
    with geo_path.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)


def numeric_case_ids(data_dir: Path) -> List[int]:
    case_ids: List[int] = []
    for geo_file in data_dir.glob("*.geo"):
        stem = geo_file.stem
        if stem.isdigit():
            case_ids.append(int(stem))
    case_ids.sort()
    if not case_ids:
        raise ValueError(f"No numeric .geo files found in {data_dir}")
    return case_ids
