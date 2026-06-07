#!/usr/bin/env python3
"""plot.py — turn a run directory into PNGs + summary.md.

Usage:
    python plot.py runs/2026-06-03_120000__dual-3090/
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bench_data


# Dark-theme, big-fonts style for OBS readability.
PLOT_STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#c9d1d9",
    "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor": "#ffffff",
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.color": "#c9d1d9",
    "ytick.color": "#c9d1d9",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    # matplotlib's legend defaults to black text — invisible on our dark
    # facecolor. Set every legend element explicitly.
    "legend.labelcolor": "#c9d1d9",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.title_fontsize": 11,
    "grid.color": "#30363d",
    "savefig.facecolor": "#0d1117",
    "savefig.edgecolor": "#0d1117",
    "savefig.dpi": 140,
}


def _apply_style() -> None:
    for k, v in PLOT_STYLE.items():
        plt.rcParams[k] = v


def _footer(fig, system: dict) -> None:
    fig.text(0.01, 0.005, bench_data.system_footer(system),
             color="#8b949e", fontsize=9, ha="left", va="bottom")


def _filter_baseline_rows(rows: list[dict]) -> list[dict]:
    """Return only c=1 steady-state rows — the single-user baseline. Used by
    per-axis plots that should show how a metric varies with ctx or gen at
    a clean single-user load; mixing c levels would average them into
    meaninglessness (e.g. decode at c=1 is 80 t/s, at c=8 is 11 t/s)."""
    return [r for r in rows
            if r.get("concurrency") == 1 and r.get("in_steady_state")]


def plot_prefill_vs_context(rows, system, out: Path) -> None:
    rows = _filter_baseline_rows(rows)
    cells = bench_data.aggregate_by_cell(rows)
    if not cells:
        return
    gen_sizes = sorted({gs for (_, gs) in cells.keys()})
    fig, ax = plt.subplots(figsize=(10, 6))
    for gs in gen_sizes:
        xs, ys, errs = [], [], []
        for (ps, g) in sorted(cells.keys()):
            if g != gs:
                continue
            c = cells[(ps, g)]
            if c["prefill_mean"] is None:
                continue
            xs.append(ps); ys.append(c["prefill_mean"]); errs.append(c["prefill_std"])
        if xs:
            ax.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2,
                        capsize=4, label=f"gen={gs}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context size (tokens)")
    ax.set_ylabel("prefill throughput (tokens/s)")
    ax.set_title("Prefill throughput vs context size")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _footer(fig, system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_throughput_vs_concurrency(rows, system, out: Path) -> None:
    """Rolling-ramp throughput plot: three panels per-cell.

    Left   — per-request decode t/s vs c (measured + fitted-model dashed).
    Middle — aggregate t/s (per-req × c) vs c (measured + fitted-model dashed).
    Right  — residual: (measured − c1/c_textbook) / c1/c_textbook vs c.

    The residual panel is the headline story for two-regime systems. It
    shows where reality deviates from the textbook c1/c expectation:
      - Negative residual = system overhead dominates (c=2..c_sat dip).
      - Zero crossing     = pure c1/c regime (well-behaved compute-bound).
      - Positive residual = parallel-decode wins (c_sat>1 boost region).
      - Returning to zero or negative at high c = compute saturation
        eventually beats parallelism.
    For a textbook compute-bound system the residual is near-zero everywhere
    and the panel is uninteresting. For anything more interesting (MTP,
    unified-memory, NPU offload) the residual exposes the regime shape
    that the headline curves only hint at."""
    cells = bench_data.aggregate_ramp_by_cell(rows)
    if not cells:
        return

    fig, (ax_pr, ax_agg, ax_res) = plt.subplots(1, 3, figsize=(20, 6))
    cmap = plt.get_cmap("tab10")
    drew_any = False
    for idx, key in enumerate(sorted(cells.keys())):
        preset, cs, gs = key
        cell = cells[key]
        c1 = cell.get("c1_decode_tps")
        fit = cell.get("fit")
        if c1 is None:
            continue
        drew_any = True
        levels = cell["levels"]
        cs_x = [lvl["c"] for lvl in levels if lvl["per_req_decode_tps"] is not None]
        per_req = [lvl["per_req_decode_tps"] for lvl in levels if lvl["per_req_decode_tps"] is not None]
        agg = [lvl["aggregate_decode_tps"] for lvl in levels if lvl["per_req_decode_tps"] is not None]
        label_base = f"{preset} ctx={cs} g={gs}"
        color = cmap(idx % 10)
        # measured (solid)
        ax_pr.plot(cs_x, per_req, marker="o", linewidth=2.2,
                   color=color, label=f"{label_base} (measured)")
        ax_agg.plot(cs_x, agg, marker="o", linewidth=2.2, color=color,
                    label=f"{label_base} (measured)")
        # Fitted hyperbolic model (T_fixed + T_per_stream*c) as dashed
        # overlay across the c range. Only draw when the fit succeeded.
        # c=1 is plotted from the fit too so the gap from measured-c=1
        # to fit-c=1 is visible — that gap quantifies how much T_fixed
        # the single-stream code path avoids vs the parallel mode.
        if fit is not None:
            max_c = max(cs_x)
            pred_x = list(range(1, max_c + 1))
            pred_per_req = [bench_data.predict_ramp_per_req(c, fit) for c in pred_x]
            pred_agg = [c * p if p is not None else None
                        for c, p in zip(pred_x, pred_per_req)]
            ceil_str = f"ceil≈{fit['aggregate_ceiling']:.0f}"
            ax_pr.plot(pred_x, pred_per_req, linestyle="--", linewidth=1.4,
                       alpha=0.7, color=color,
                       label=f"{label_base} (fit, {ceil_str} t/s)")
            ax_agg.plot(pred_x, pred_agg, linestyle="--", linewidth=1.4,
                        alpha=0.7, color=color)

        # Efficiency peak marker — vertical line at the c where aggregate
        # maxes for this cell. Past this c, adding concurrency reduces total
        # work done (per-stream loss > additional parallelism). Operator-
        # facing translation: "run with N slots for max throughput on this
        # cell; more than that and you're hurting yourself." Dotted so it
        # doesn't compete with the measured/predicted curves visually.
        if agg:
            peak_idx = agg.index(max(agg))
            peak_c = cs_x[peak_idx]
            ax_agg.axvline(peak_c, color=color, linestyle=":", linewidth=1.5,
                           alpha=0.55)
            # Annotate the marker line near the top of the aggregate axis
            # with the c value, so the chart is self-explanatory at a glance.
            ax_agg.annotate(f"peak c={peak_c}", xy=(peak_c, max(agg)),
                            xytext=(3, 5), textcoords="offset points",
                            fontsize=7, color=color, alpha=0.85)

        # Residual against the textbook c1/c expectation (NOT the fitted
        # hyperbolic model — using the fit would zero out the deviation
        # signal we want to surface, since the fit is built FROM the
        # measurements). Measured points only; no fit curve because the
        # residual IS the relationship to the prediction.
        residual_pct = [100.0 * (pr - c1 / c) / (c1 / c)
                        for c, pr in zip(cs_x, per_req)]
        ax_res.plot(cs_x, residual_pct, marker="o", linewidth=2.2,
                    color=color, label=label_base)

    if not drew_any:
        plt.close(fig)
        return

    ax_pr.set_xlabel("concurrency (target N-equivalent sustained users)")
    ax_pr.set_ylabel("per-request decode t/s")
    ax_pr.set_title("Per-request throughput vs concurrency")
    ax_pr.set_ylim(bottom=0)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend(loc="best", fontsize=8)
    ax_agg.set_xlabel("concurrency")
    ax_agg.set_ylabel("aggregate decode t/s (per-req × c)")
    ax_agg.set_title("Aggregate throughput vs concurrency")
    ax_agg.set_ylim(bottom=0)
    ax_agg.grid(True, alpha=0.3)
    ax_res.set_xlabel("concurrency")
    ax_res.set_ylabel("residual vs textbook c1/c (%)")
    ax_res.set_title("Residual — where reality diverges from c1/c")
    ax_res.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax_res.grid(True, alpha=0.3)
    ax_res.legend(loc="best", fontsize=8)
    _footer(fig, system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_ttft_vs_concurrency(rows, system, out: Path) -> None:
    """TTFT distribution by concurrency level — shows how TTFT grows as the
    in-flight cap throttles fires. Uses steady-state samples only so the
    transient ramp-up to N in-flight doesn't bias the mean."""
    from collections import defaultdict
    by_n: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if (r.get("ttft_ms") is not None and r.get("concurrency")
                and r.get("in_steady_state")):
            by_n[r["concurrency"]].append(r["ttft_ms"])
    if len(by_n) < 2:
        return
    levels = sorted(by_n)
    means = [sum(by_n[n]) / len(by_n[n]) for n in levels]
    # P95: tail-latency story
    def p95(xs):
        s = sorted(xs)
        return s[int(0.95 * (len(s) - 1))] if s else 0
    p95s = [p95(by_n[n]) for n in levels]
    mins = [min(by_n[n]) for n in levels]
    maxs = [max(by_n[n]) for n in levels]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(levels, means, marker="o", linewidth=2.5, label="TTFT mean",
            color="#a5d6ff")
    ax.plot(levels, p95s, marker="^", linewidth=2, label="TTFT p95",
            linestyle="--", color="#ffd166")
    ax.fill_between(levels, mins, maxs, alpha=0.15, color="#a5d6ff", label="min/max")
    ax.set_xticks(levels)
    ax.set_xlabel("concurrency (parallel requests)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT vs concurrency")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _footer(fig, system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_ttft_vs_context(rows, system, out: Path) -> None:
    """TTFT (time-to-first-token) vs context size at c=1 baseline — dominated
    by prefill cost, so should scale ~linearly with ctx. Mostly insensitive
    to gen_size since we measure to the first chunk."""
    rows = _filter_baseline_rows(rows)
    cells = bench_data.aggregate_by_cell(rows)
    if not cells or not any(c.get("ttft_ms_mean") is not None for c in cells.values()):
        return
    gen_sizes = sorted({gs for (_, gs) in cells.keys()})
    fig, ax = plt.subplots(figsize=(10, 6))
    for gs in gen_sizes:
        xs, ys, errs = [], [], []
        for (ps, g) in sorted(cells.keys()):
            if g != gs:
                continue
            c = cells[(ps, g)]
            if c.get("ttft_ms_mean") is None:
                continue
            xs.append(ps); ys.append(c["ttft_ms_mean"]); errs.append(c.get("ttft_ms_std", 0))
        if xs:
            ax.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2,
                        capsize=4, label=f"gen={gs}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context size (tokens)")
    ax.set_ylabel("time to first token (ms)")
    ax.set_title("TTFT vs context size")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")
    _footer(fig, system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_decode_vs_gen(rows, system, out: Path) -> None:
    """Decode throughput vs gen_size at c=1 baseline. Should be roughly flat
    once gen is long enough to amortize MTP/draft setup. Multi-c story is
    in the ramp plot."""
    rows = _filter_baseline_rows(rows)
    cells = bench_data.aggregate_by_cell(rows)
    if not cells:
        return
    context_sizes = sorted({ps for (ps, _) in cells.keys()})
    fig, ax = plt.subplots(figsize=(10, 6))
    for ps in context_sizes:
        xs, ys, errs = [], [], []
        for (p, gs) in sorted(cells.keys()):
            if p != ps:
                continue
            c = cells[(p, gs)]
            if c["decode_mean"] is None:
                continue
            xs.append(gs); ys.append(c["decode_mean"]); errs.append(c["decode_std"])
        if xs:
            ax.errorbar(xs, ys, yerr=errs, marker="s", linewidth=2,
                        capsize=4, label=f"ctx={ps}")
    ax.set_xlabel("generation length (tokens)")
    ax.set_ylabel("decode throughput (tokens/s)")
    ax.set_title("Decode throughput vs generation length")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _footer(fig, system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_decode(rows, system, out: Path) -> None:
    """Decode throughput heatmap (ctx × gen) at c=1 baseline. The c-axis
    is in the ramp plot — this view is just the single-user baseline shape.
    Skipped when fewer than 2 ctx or 2 gen values: a single tile shows
    nothing."""
    rows = _filter_baseline_rows(rows)
    cells = bench_data.aggregate_by_cell(rows)
    if not cells:
        return
    context_sizes = sorted({ps for (ps, _) in cells.keys()})
    gen_sizes = sorted({gs for (_, gs) in cells.keys()})
    if len(context_sizes) < 2 and len(gen_sizes) < 2:
        return
    Z = np.full((len(gen_sizes), len(context_sizes)), np.nan)
    for i, gs in enumerate(gen_sizes):
        for j, ps in enumerate(context_sizes):
            c = cells.get((ps, gs))
            if c and c["decode_mean"] is not None:
                Z[i, j] = c["decode_mean"]
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(Z, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(context_sizes)), [str(p) for p in context_sizes])
    ax.set_yticks(range(len(gen_sizes)), [str(g) for g in gen_sizes])
    ax.set_xlabel("context size (tokens)")
    ax.set_ylabel("generation length (tokens)")
    ax.set_title("Decode throughput (t/s)")
    for i in range(len(gen_sizes)):
        for j in range(len(context_sizes)):
            v = Z[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v < np.nanmean(Z) else "black", fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("decode t/s")
    _footer(fig, system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_draft_accept(rows, system, out: Path) -> bool:
    """Draft acceptance per request + drafted-vs-accepted histogram. Uses
    steady-state rows so transient ramp-up noise isn't included."""
    draft_rows = [r for r in rows if r.get("draft_n") and r.get("in_steady_state")]
    if not draft_rows:
        return False
    accept_rates = [
        (r["draft_accepted"] / r["draft_n"]) for r in draft_rows
        if r.get("draft_accepted") is not None and r["draft_n"]
    ]
    if not accept_rates:
        return False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.bar(range(len(accept_rates)), [r * 100 for r in accept_rates],
            color="#bc8cff", edgecolor="#7c3aed")
    ax1.axhline(mean(accept_rates) * 100, linestyle="--", color="#f59e0b",
                label=f"mean {mean(accept_rates)*100:.1f}%")
    ax1.set_xlabel("request #")
    ax1.set_ylabel("draft acceptance (%)")
    ax1.set_title("Draft acceptance per request")
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    drafts = [r["draft_n"] for r in draft_rows]
    accepts = [r["draft_accepted"] for r in draft_rows if r.get("draft_accepted") is not None]
    bins = np.linspace(0, max(drafts + accepts + [1]), 20)
    ax2.hist(drafts, bins=bins, alpha=0.6, label="drafted", color="#60a5fa")
    ax2.hist(accepts, bins=bins, alpha=0.6, label="accepted", color="#34d399")
    ax2.set_xlabel("tokens per request")
    ax2.set_ylabel("count")
    ax2.set_title("Drafted vs accepted distribution")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    _footer(fig, system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_gpu_timeline(telemetry_csv: Path, system, out: Path) -> bool:
    if not telemetry_csv.exists():
        return False
    by_gpu: dict[int, dict[str, list]] = {}
    with telemetry_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["index"])
            except (ValueError, KeyError):
                continue
            d = by_gpu.setdefault(idx, {"t": [], "util": [], "power": [], "vram": []})
            d["t"].append(row.get("timestamp"))
            try:
                d["util"].append(float(row.get("util_gpu") or "nan"))
            except ValueError:
                d["util"].append(float("nan"))
            try:
                d["power"].append(float(row.get("power_w") or "nan"))
            except ValueError:
                d["power"].append(float("nan"))
            try:
                d["vram"].append(float(row.get("vram_used_mib") or "nan") / 1024.0)
            except ValueError:
                d["vram"].append(float("nan"))
    if not by_gpu:
        return False
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for idx, d in sorted(by_gpu.items()):
        xs = list(range(len(d["t"])))
        axes[0].plot(xs, d["util"], label=f"gpu{idx}", linewidth=2)
        axes[1].plot(xs, d["power"], label=f"gpu{idx}", linewidth=2)
        axes[2].plot(xs, d["vram"], label=f"gpu{idx}", linewidth=2)
    axes[0].set_ylabel("util %"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("power (W)"); axes[1].grid(True, alpha=0.3)
    axes[2].set_ylabel("VRAM used (GiB)"); axes[2].set_xlabel("sample #")
    axes[2].grid(True, alpha=0.3)
    axes[0].set_title("GPU telemetry over sweep")
    _footer(fig, system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def write_summary_md(rows, system, summary: dict, out: Path) -> None:
    """Write the headline summary.md for a rolling-ramp run.

    The headline numbers are now ramp-aware:
      - c=1 baseline decode (single-user experience)
      - aggregate peak t/s and the c where it occurred
      - per-req threshold c (first c < 50% of c=1)

    The detailed per-cell ramp tables come from write_summary_md_multi which
    appends after this function.
    """
    lines: list[str] = []
    lines.append(f"# {(system.get('config') or {}).get('name', 'benchmark')} — summary")
    lines.append("")
    lines.append("## System")
    lines.append("")
    lines.append(f"- **rig**: {system.get('rig_label')}")
    lines.append(f"- **host**: {(system.get('host') or {}).get('hostname')} "
                 f"({(system.get('host') or {}).get('distro')}, kernel "
                 f"{(system.get('host') or {}).get('kernel')})")
    lines.append(f"- **cpu**: {(system.get('cpu') or {}).get('model')}")
    lines.append(f"- **ram**: {(system.get('memory') or {}).get('total_gb')} GiB")
    llc = system.get("llama_cpp") or {}
    lines.append(f"- **llama.cpp**: {llc.get('git_commit')} "
                 f"(flags: {llc.get('build_flags')})")
    lines.append("- **gpus**:")
    sel = (system.get("selection") or {}).get("cuda_visible_devices")
    sel_set = {s.strip() for s in str(sel).split(",")} if sel else None
    for g in system.get("gpus") or []:
        marker = " ← selected" if (sel_set and str(g.get("index")) in sel_set) else ""
        lines.append(f"  - gpu{g.get('index')}: {g.get('name')} "
                     f"({(g.get('vram_total_mib') or 0)/1024:.0f} GiB, "
                     f"drv {g.get('driver')}, CUDA {g.get('cuda')}){marker}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")

    # Ramp-aware headline: pick the c=1 baseline + aggregate peak from each cell.
    ramp_cells = bench_data.aggregate_ramp_by_cell(rows)
    if ramp_cells:
        lines.append("| cell | c=1 t/s | aggregate peak | peak @ c | <50% c=1 @ c |")
        lines.append("|---|---:|---:|---:|---:|")
        for key in sorted(ramp_cells.keys()):
            preset, cs, gs = key
            cell = ramp_cells[key]
            c1 = cell.get("c1_decode_tps")
            peak_c = cell.get("aggregate_peak_c")
            thresh_c = cell.get("per_req_threshold_c")
            agg_at_peak = (next((lvl["aggregate_decode_tps"] for lvl in cell["levels"]
                                 if lvl["c"] == peak_c), None) if peak_c else None)
            lines.append(
                f"| `{preset}` ctx={cs} gen={gs} | "
                f"{f'**{c1:.1f}**' if c1 else '—'} | "
                f"{f'{agg_at_peak:.1f}' if agg_at_peak is not None else '—'} | "
                f"{peak_c if peak_c is not None else '—'} | "
                f"{thresh_c if thresh_c is not None else '—'} |"
            )
    lines.append("")

    # Operator recommendation per cell (#27): does concurrency win on
    # this rig for this workload, or should the operator serve
    # sequentially? Classify each cell by ceiling-vs-c1 ratio and
    # group into actionable buckets. Critical content for the shared
    # artifact — this is what someone posting to r/LocalLlama or
    # discussing on Discord will want to point at.
    if ramp_cells:
        wins, marg, loses = [], [], []
        for key in sorted(ramp_cells.keys()):
            preset, cs, gs = key
            cell = ramp_cells[key]
            regime = bench_data.classify_cell_regime(cell)
            if regime is None:
                continue
            label = f"`{preset}` ctx={cs} gen={gs}"
            entry = (label, regime)
            if regime["regime"] == "concurrency_wins":
                wins.append(entry)
            elif regime["regime"] == "sequential_wins":
                loses.append(entry)
            else:
                marg.append(entry)
        if wins or loses or marg:
            lines.append("## Operator recommendation")
            lines.append("")
            lines.append(
                "Per-cell verdict: does serving N concurrent users actually "
                "increase **aggregate** throughput vs serving them sequentially? "
                "Compares the cell's hyperbolic-fit aggregate ceiling against "
                "the c=1 baseline. **For cells where the ceiling is below the "
                "c=1 baseline, concurrency hurts total throughput** — queue and "
                "serve one at a time, or pick different hardware for that "
                "workload."
            )
            lines.append("")
        if loses:
            lines.append("**Sequential wins** — concurrency reduces aggregate throughput:")
            for label, r in loses:
                lines.append(f"- {label} — ceiling = {r['ratio']*100:.0f}% of c=1")
            lines.append("")
        if wins:
            lines.append("**Concurrency wins** — serve concurrent users for more total t/s:")
            for label, r in wins:
                lines.append(f"- {label} — ceiling = {r['ratio']*100:.0f}% of c=1 ({r['label']})")
            lines.append("")
        if marg:
            lines.append("**Marginal** — concurrency neither helps nor hurts much:")
            for label, r in marg:
                lines.append(f"- {label} — ceiling = {r['ratio']*100:.0f}% of c=1")
            lines.append("")

    out.write_text("\n".join(lines))
    return


def plot_overlay_by_preset(rows, system, out: Path, *,
                           metric: str, x_key: str,
                           x_label: str, y_label: str, title: str,
                           x_log: bool = False) -> None:
    """One line per preset, x = gen (or ctx), y = metric mean.

    Collapses across the other size axis (a sensible "single number per preset
    per x" overlay for visual diff).
    """
    presets = sorted({r.get("preset") for r in rows if r.get("preset")})
    if len(presets) < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for preset in presets:
        sub = [r for r in rows if r.get("preset") == preset]
        cells = bench_data.aggregate_by_cell(sub)
        if not cells:
            continue
        x_vals = sorted({(gs if x_key == "gen_size" else cs)
                         for (cs, gs) in cells.keys()})
        ys, errs = [], []
        for xv in x_vals:
            keyed = [v for k, v in cells.items()
                     if (k[1] if x_key == "gen_size" else k[0]) == xv]
            metric_vals = [c[metric] for c in keyed if c[metric] is not None]
            if metric_vals:
                ys.append(mean(metric_vals))
                errs.append(0)
            else:
                ys.append(float("nan"))
                errs.append(0)
        ax.plot(x_vals, ys, marker="o", linewidth=2, label=preset)
    if x_log:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", title="preset")
    _footer(fig, system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_summary_md_multi(rows, system, summary: dict, out: Path) -> None:
    """Extended summary.md — per-cell ramp tables with hyperbolic fit overlay.

    Each (preset, ctx, gen) cell gets one ramp table showing every measured
    c level with predicted t/s (from the T_fixed + T_per_stream hyperbolic
    fit) and deviation. Below each table: knee labels and stop reason."""
    write_summary_md(rows, system, summary, out)

    cells = bench_data.aggregate_ramp_by_cell(rows)
    if not cells:
        return

    lines = [out.read_text().rstrip(), ""]
    lines.append("## Per-cell ramp breakdown")
    lines.append("")
    lines.append("Each table shows the measured ramp from c=1 to the stop "
                 "point. **Predicted** is the fit `per_req(c) = 1/(T_fixed + "
                 "T_per_stream·c)` — a physically-motivated model where each "
                 "decode step pays a fixed batch overhead plus per-stream cost. "
                 "Fit is on c≥2 only (c=1 is a separate code path). **Δ** is "
                 "`measured/predicted-1` — small Δ means the system follows the "
                 "model cleanly; large Δ flags a kernel quirk, thermal, memory "
                 "pressure, or other non-textbook behavior. `T_per_stream` "
                 "directly implies an aggregate ceiling of `1/T_per_stream` t/s.")
    lines.append("")

    for key in sorted(cells.keys()):
        preset, cs, gs = key
        cell = cells[key]
        c1 = cell.get("c1_decode_tps")
        fit = cell.get("fit")
        peak_c = cell.get("aggregate_peak_c")
        thresh_c = cell.get("per_req_threshold_c")
        if c1 is None:
            lines.append(f"### `{preset}` ctx={cs} gen={gs}")
            lines.append("")
            lines.append("_no c=1 baseline — ramp aborted_")
            lines.append("")
            continue
        labels = [f"c1={c1:.1f} t/s"]
        if fit is not None:
            labels.append(
                f"fit: T_fixed={fit['t_fixed']*1000:.2f} ms/tok, "
                f"T_per_stream={fit['t_per_stream']*1000:.2f} ms/tok, "
                f"ceiling≈{fit['aggregate_ceiling']:.1f} t/s"
            )
        if peak_c is not None:
            agg_at_peak = next((lvl["aggregate_decode_tps"] for lvl in cell["levels"]
                                if lvl["c"] == peak_c), None)
            labels.append(
                f"aggregate peak@c={peak_c} ({agg_at_peak:.1f} t/s)"
                if agg_at_peak is not None else f"aggregate peak@c={peak_c}")
        if thresh_c is not None:
            labels.append(f"<50% c=1@c={thresh_c}")

        lines.append(f"### `{preset}` ctx={cs} gen={gs}")
        lines.append("")
        lines.append(" · ".join(labels))
        lines.append("")
        lines.append("| c | per-req t/s | predicted t/s | Δ | aggregate t/s | "
                     "prefill t/s | TTFT mean ms | TTFT p95 ms | accept % | n |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for lvl in cell["levels"]:
            pr = lvl.get("per_req_decode_tps")
            pred = lvl.get("predicted_per_req")
            dev = lvl.get("deviation")
            agg = lvl.get("aggregate_decode_tps")
            pre = lvl.get("prefill_tps_tw")
            ttm = lvl.get("ttft_ms_mean")
            ttp = lvl.get("ttft_ms_p95")
            acc = lvl.get("accept_mean")
            lines.append(
                f"| {lvl['c']} | "
                f"{f'**{pr:.1f}**' if pr is not None else '-'} | "
                f"{f'{pred:.1f}' if pred is not None else '-'} | "
                f"{f'{dev*100:+.1f}%' if dev is not None else '-'} | "
                f"{f'{agg:.1f}' if agg is not None else '-'} | "
                f"{f'{pre:.0f}' if pre is not None else '-'} | "
                f"{f'{ttm:.0f}' if ttm is not None else '-'} | "
                f"{f'{ttp:.0f}' if ttp is not None else '-'} | "
                f"{f'{acc*100:.1f}' if acc is not None else '-'} | "
                f"{lvl.get('n_steady', 0)} |"
            )
        lines.append("")

    out.write_text("\n".join(lines))


def _emit_cell_table(lines: list[str], cells: dict) -> None:
    lines.append("| ctx | gen | prefill t/s | decode t/s | ms/tok | TTFT ms | accept % | n |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (cs, gs) in sorted(cells.keys()):
        c = cells[(cs, gs)]
        prefill = (f"{c['prefill_mean']:.1f} ±{c['prefill_std']:.1f}"
                   if c.get("prefill_mean") is not None else "-")
        decode = (f"{c['decode_mean']:.1f} ±{c['decode_std']:.1f}"
                  if c.get("decode_mean") is not None else "-")
        mstok = (f"{c['decode_ms_per_token']:.2f}"
                 if c.get("decode_ms_per_token") is not None else "-")
        ttft = (f"{c['ttft_ms_mean']:.0f}"
                if c.get("ttft_ms_mean") is not None else "-")
        accept = (f"{c['accept_mean']*100:.1f}"
                  if c.get("accept_mean") is not None else "-")
        lines.append(f"| {cs} | {gs} | {prefill} | {decode} | {mstok} | "
                     f"{ttft} | {accept} | {c.get('n', 0)} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    _apply_style()

    run_dir: Path = args.run_dir
    rows = bench_data.load_jsonl(run_dir / "raw.jsonl")
    system = bench_data.load_system(run_dir)
    if not rows:
        print(f"no rows in {run_dir}/raw.jsonl", file=__import__("sys").stderr)
        return 1

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    presets = sorted({r["preset"] for r in rows if r.get("preset")})

    if len(presets) == 1:
        # Single-preset path — flat filenames, matches pre-preset behavior.
        plot_prefill_vs_context(rows, system, plots_dir / "prefill_vs_context.png")
        plot_decode_vs_gen(rows, system, plots_dir / "decode_vs_gen.png")
        plot_ttft_vs_context(rows, system, plots_dir / "ttft_vs_context.png")
        plot_heatmap_decode(rows, system, plots_dir / "heatmap_decode.png")
        plot_draft_accept(rows, system, plots_dir / "draft_accept_rate.png")
    else:
        # Multi-preset: one set of facet PNGs per preset + overlay PNGs.
        for preset in presets:
            sub = [r for r in rows if r["preset"] == preset]
            safe = "".join(c if c.isalnum() else "_" for c in preset)
            plot_prefill_vs_context(sub, system, plots_dir / f"prefill_vs_context__{safe}.png")
            plot_decode_vs_gen(sub, system, plots_dir / f"decode_vs_gen__{safe}.png")
            plot_ttft_vs_context(sub, system, plots_dir / f"ttft_vs_context__{safe}.png")
            plot_heatmap_decode(sub, system, plots_dir / f"heatmap_decode__{safe}.png")
            plot_draft_accept(sub, system, plots_dir / f"draft_accept_rate__{safe}.png")
        # Overlays: per-preset on a single chart.
        plot_overlay_by_preset(rows, system,
                               plots_dir / "decode_vs_gen_by_preset.png",
                               metric="decode_mean", x_key="gen_size",
                               x_label="generation length (tokens)",
                               y_label="decode t/s (mean across ctx)",
                               title="Decode throughput by preset")
        plot_overlay_by_preset(rows, system,
                               plots_dir / "prefill_vs_context_by_preset.png",
                               metric="prefill_mean", x_key="context_size",
                               x_label="context size (tokens)",
                               y_label="prefill t/s (mean across gen)",
                               title="Prefill throughput by preset", x_log=True)

    # Concurrency charts — only emit when the run sampled multiple concurrency
    # levels (otherwise the chart is a single point, uninformative).
    n_concurrency = len({r.get("concurrency") for r in rows if r.get("concurrency")})
    if n_concurrency > 1:
        plot_throughput_vs_concurrency(rows, system,
                                       plots_dir / "throughput_vs_concurrency.png")
        plot_ttft_vs_concurrency(rows, system,
                                 plots_dir / "ttft_vs_concurrency.png")

    plot_gpu_timeline(run_dir / "gpu_telemetry.csv", system,
                      plots_dir / "gpu_timeline.png")

    summary = bench_data.load_summary(run_dir)
    write_summary_md_multi(rows, system, summary, run_dir / "summary.md")
    print(f"plots → {plots_dir}  ({len(presets)} preset(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
