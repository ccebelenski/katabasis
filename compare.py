#!/usr/bin/env python3
"""compare.py — A/B (or N-way) comparison across run directories.

Usage:
    python compare.py runs/A runs/B [runs/C ...] [-o comparison/]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bench_data
from plot import PLOT_STYLE, _apply_style, _footer


@dataclass
class Loaded:
    run_dir: Path
    system: dict
    rows: list[dict]
    cells: dict
    label: str


def load(run_dir: Path) -> Loaded:
    system = bench_data.load_system(run_dir)
    rows = bench_data.load_jsonl(run_dir / "raw.jsonl")
    cells = bench_data.aggregate_by_cell(rows)
    idx = bench_data.index_run(run_dir)
    if idx:
        label = f"{idx.model}/{idx.quant} @ {idx.rig_label}"
    else:
        label = run_dir.name
    return Loaded(run_dir, system, rows, cells, label)


def configs_equivalent(loads: list[Loaded]) -> bool:
    """Are sweep configs identical? Different configs → no single speedup number."""
    seen = set()
    for l in loads:
        cfg = (l.system.get("config") or {}).get("sweep", {})
        seen.add((tuple(bench_data.get_context_sizes(cfg)),
                  tuple(cfg.get("gen_sizes") or []),
                  tuple(bench_data.get_prompt_presets(cfg))))
    return len(seen) <= 1


def rigs_equivalent(loads: list[Loaded]) -> bool:
    return len({l.system.get("rig_label") for l in loads}) <= 1


def plot_grouped_bars(loads: list[Loaded], metric: str, out: Path,
                      ylabel: str, title: str) -> None:
    """Grouped bars per (context_size, gen_size) cell."""
    all_keys = sorted({k for l in loads for k in l.cells.keys()})
    if not all_keys:
        return
    n = len(loads)
    width = 0.8 / n
    x = np.arange(len(all_keys))
    fig, ax = plt.subplots(figsize=(max(10, len(all_keys) * 0.8), 6))
    for i, l in enumerate(loads):
        ys = []
        for k in all_keys:
            c = l.cells.get(k, {})
            v = c.get(metric)
            ys.append(v if v is not None else 0)
        ax.bar(x + (i - (n - 1) / 2) * width, ys, width, label=l.label)
    ax.set_xticks(x, [f"ctx={p}\ng={g}" for (p, g) in all_keys], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _footer(fig, loads[0].system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_decode_vs_gen(loads: list[Loaded], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for l in loads:
        context_sizes = sorted({cs for (cs, _) in l.cells.keys()})
        # Mean across all context sizes per gen size.
        gen_sizes = sorted({gs for (_, gs) in l.cells.keys()})
        ys = []
        for gs in gen_sizes:
            vals = [l.cells[(cs, gs)]["decode_mean"] for cs in context_sizes
                    if (cs, gs) in l.cells and l.cells[(cs, gs)]["decode_mean"] is not None]
            ys.append(mean(vals) if vals else 0)
        ax.plot(gen_sizes, ys, marker="o", linewidth=2, label=l.label)
    ax.set_xlabel("generation length (tokens)")
    ax.set_ylabel("decode t/s (mean across context sizes)")
    ax.set_title("Decode throughput overlay")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _footer(fig, loads[0].system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def compute_speedup(loads: list[Loaded]) -> float | None:
    """Speedup of loads[0] vs loads[1], averaged across common cells."""
    if len(loads) != 2:
        return None
    common = set(loads[0].cells.keys()) & set(loads[1].cells.keys())
    if not common:
        return None
    ratios = []
    for k in common:
        a = loads[0].cells[k].get("decode_mean")
        b = loads[1].cells[k].get("decode_mean")
        if a and b and b > 0:
            ratios.append(a / b)
    return mean(ratios) if ratios else None


def write_diff_md(loads: list[Loaded], out: Path, speedup: float | None,
                  speedup_safe: bool) -> None:
    lines: list[str] = []
    lines.append("# Comparison")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    for l in loads:
        lines.append(f"- **{l.label}**  ({l.run_dir})")
    lines.append("")

    if len(loads) == 2:
        lines.append("## System diff (A → B)")
        lines.append("")
        diff = bench_data.system_diff(loads[0].system, loads[1].system)
        if not diff:
            lines.append("_no differences captured in system manifest_")
        else:
            lines.append("| field | A | B |")
            lines.append("|---|---|---|")
            for f, va, vb in diff:
                lines.append(f"| {f} | {va} | {vb} |")
        lines.append("")
        lines.append("## Headline")
        lines.append("")
        if speedup is None:
            lines.append("_(insufficient overlap to compute speedup)_")
        elif not speedup_safe:
            lines.append(f"⚠️ **Apples-to-oranges**: rig and sweep config both differ. "
                         f"Raw decode ratio (A/B): {speedup:.2f}× — not a meaningful speedup.")
        else:
            lines.append(f"**Decode speedup (A/B)**: **{speedup:.2f}×**  "
                         f"(mean across {len(set(loads[0].cells.keys()) & set(loads[1].cells.keys()))} "
                         f"shared cells)")
    out.write_text("\n".join(lines))


def compare(run_dirs: list[Path], out_dir: Path) -> None:
    _apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    loads = [load(p) for p in run_dirs]

    plot_grouped_bars(loads, "decode_mean", out_dir / "decode_bars.png",
                      "decode t/s", "Decode throughput per (prompt, gen) cell")
    plot_grouped_bars(loads, "prefill_mean", out_dir / "prefill_bars.png",
                      "prefill t/s", "Prefill throughput per (prompt, gen) cell")
    plot_overlay_decode_vs_gen(loads, out_dir / "decode_overlay.png")

    speedup = compute_speedup(loads)
    safe = configs_equivalent(loads) and (rigs_equivalent(loads) or
                                          # treat same-rig+different-config as still meaningful
                                          False)
    # More forgiving rule: speedup is "safe" when configs match.
    # Cross-rig comparisons remain labeled but still get a ratio.
    safe = configs_equivalent(loads)

    write_diff_md(loads, out_dir / "summary.md", speedup, safe)
    print(f"comparison → {out_dir}")
    if speedup is not None:
        tag = "speedup" if safe else "ratio (apples-to-oranges)"
        print(f"{tag}: {speedup:.2f}x")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("comparison"))
    args = ap.parse_args()
    compare(args.run_dirs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
