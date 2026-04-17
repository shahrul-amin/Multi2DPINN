from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import savemat

from data_gen.common import numeric_case_ids, parse_rectangles, segment_centerline_from_rectangle

from src.first_stage.config import Omega, Z, e, rou

Point = Tuple[float, float]


def rounded_point(point: Point) -> Point:
    return (round(point[0], 6), round(point[1], 6))


def build_node_traces(node_count: int, t_norm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    traces = np.zeros((node_count, t_norm.size), dtype=np.float64)
    for idx in range(node_count):
        amp = rng.uniform(-8.0e6, 8.0e6)
        tau = rng.uniform(0.15, 0.8)
        freq = int(rng.integers(1, 4))
        phase = rng.uniform(0.0, 2.0 * np.pi)
        traces[idx] = amp * (1.0 - np.exp(-t_norm / tau)) * np.sin(phase + 2.0 * np.pi * freq * t_norm)
    return traces


def simulate_case(
    geo_path: Path,
    mat_path: Path,
    j_min: float,
    j_max: float,
    time_points: int,
    seed: int,
) -> None:
    rectangles = parse_rectangles(geo_path)

    segments: List[Tuple[Point, Point, float, float]] = []
    for rect in rectangles:
        start, end, _ = segment_centerline_from_rectangle(rect)
        length_um = max(abs(rect[2]), abs(rect[3]))
        length_m = length_um * 1.0e-6
        segments.append((rounded_point(start), rounded_point(end), length_um, length_m))

    node_index: Dict[Point, int] = {}
    for start, end, _, _ in segments:
        if start not in node_index:
            node_index[start] = len(node_index)
        if end not in node_index:
            node_index[end] = len(node_index)

    rng = np.random.default_rng(seed)
    t_norm = np.linspace(0.0, 1.0, num=time_points, endpoint=True)
    node_traces = build_node_traces(len(node_index), t_norm, rng)

    j_values = rng.uniform(j_min, j_max, size=len(segments)).astype(np.float64)
    svc = np.empty((len(segments), 1), dtype=object)

    for seg_idx, (start, end, length_um, length_m) in enumerate(segments):
        n_x = max(int(round(length_um)) + 1, 5)
        s = np.linspace(0.0, 1.0, num=n_x, endpoint=True)
        s_col = s[:, None]

        start_trace = node_traces[node_index[start]]
        end_trace = node_traces[node_index[end]]
        base = (1.0 - s_col) * start_trace[None, :] + s_col * end_trace[None, :]

        g_val = e * Z * rou * j_values[seg_idx] / Omega
        tau = rng.uniform(0.2, 0.9)
        temporal = (1.0 - np.exp(-t_norm / tau))[None, :]

        gain_curve = rng.uniform(0.02, 0.05)
        curve = gain_curve * g_val * length_m * (s * (1.0 - s))[:, None] * temporal

        gain_osc = rng.uniform(0.005, 0.015)
        phase_osc = rng.uniform(0.0, 2.0 * np.pi)
        osc = gain_osc * g_val * length_m * np.sin(np.pi * s)[:, None] * np.sin(2.0 * np.pi * t_norm + phase_osc)[
            None, :
        ]

        svc[seg_idx, 0] = (base + curve + osc).astype(np.float64)

    savemat(mat_path, {"J": j_values.reshape(1, -1), "sVC": svc})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate .mat simulation files from .geo")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--start-id", type=int)
    parser.add_argument("--end-id", type=int)
    parser.add_argument("--j-min", type=float, default=1.0e9)
    parser.add_argument("--j-max", type=float, default=5.0e9)
    parser.add_argument("--time-points", type=int, default=101)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.j_min <= 0.0 or args.j_max <= 0.0:
        raise ValueError("j-min and j-max must be positive")
    if args.j_min >= args.j_max:
        raise ValueError("j-min must be smaller than j-max")
    if args.time_points < 2:
        raise ValueError("time-points must be >= 2")

    if args.start_id is None and args.end_id is None:
        case_ids = numeric_case_ids(args.data_dir)
    elif args.start_id is not None and args.end_id is not None:
        if args.start_id > args.end_id:
            raise ValueError("start-id must be <= end-id")
        case_ids = list(range(args.start_id, args.end_id + 1))
    else:
        raise ValueError("start-id and end-id must be set together")

    for case_id in case_ids:
        geo_path = args.data_dir / f"{case_id}.geo"
        mat_path = args.data_dir / f"{case_id}.mat"
        if not geo_path.exists():
            raise FileNotFoundError(f"{geo_path} does not exist")
        if mat_path.exists() and not args.overwrite:
            raise FileExistsError(f"{mat_path} already exists")

        simulate_case(
            geo_path=geo_path,
            mat_path=mat_path,
            j_min=args.j_min,
            j_max=args.j_max,
            time_points=args.time_points,
            seed=args.seed + case_id,
        )

    print(f"generated {len(case_ids)} .mat files in {args.data_dir}")


if __name__ == "__main__":
    main()
