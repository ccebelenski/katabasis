"""Shared data-loading helpers for katabasis / plot / compare / weatherman."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any


@dataclass
class RunIndex:
    """One row in the weatherman index — distilled summary of a run dir."""
    path: Path
    timestamp: str
    rig_label: str
    model: str
    quant: str
    draft_model: str | None
    spec_type: str | None
    presets: list[str]
    gpus_selected: str
    commit: str | None
    decode_tps_mean: float | None
    prefill_tps_mean: float | None
    ttft_ms_mean: float | None
    accept_rate_mean: float | None
    rows: int


# ---------- visual helpers (shared with katabasis / weatherman) --------------

SPARK_BLOCKS = "▁▂▃▄▅▆▇█"

# Braille dot positions inside a single cell (col, row_from_top) → bit:
#   1 4         dot1 = 0x01  dot4 = 0x08
#   2 5         dot2 = 0x02  dot5 = 0x10
#   3 6         dot3 = 0x04  dot6 = 0x20
#   7 8         dot7 = 0x40  dot8 = 0x80
_BRAILLE_BITS = {
    (0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
    (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80,
}

_GPU_NAME_STRIP = (
    "NVIDIA GeForce ", "NVIDIA ", "Generation",
    "Workstation Edition", "Max-Q ",
)


def short_gpu_name(name: str | None) -> str:
    s = str(name or "?")
    for tok in _GPU_NAME_STRIP:
        s = s.replace(tok, "")
    return " ".join(s.split())


def spark(values, width: int,
          lo: float | None = None, hi: float | None = None) -> str:
    """One-row unicode-block sparkline. Right-aligns trailing samples."""
    vals = [v for v in values if v is not None]
    if not vals:
        return " " * width
    vals = vals[-width:]
    lo = min(vals) if lo is None else min(lo, min(vals))
    hi = max(vals) if hi is None else max(hi, max(vals))
    if hi <= lo:
        hi = lo + 1.0
    out = []
    for v in vals:
        i = int(round((v - lo) / (hi - lo) * (len(SPARK_BLOCKS) - 1)))
        out.append(SPARK_BLOCKS[max(0, min(len(SPARK_BLOCKS) - 1, i))])
    return " " * (width - len(out)) + "".join(out)


def time_bucketed_braille_bars(
    samples: "list[tuple[float, float]]",
    width: int, height: int,
    t_start: float, t_end: float,
    lo: float | None = None, hi: float | None = None,
) -> list[str]:
    """Render timestamped samples as a Braille bar chart with a SHARED time
    axis. All metrics rendered with the same (t_start, t_end) and width
    line up column-for-column wall-clock-wise — letting a viewer correlate
    e.g. GPU power spikes with decode-rate drops at the same instant.

    samples: list of (timestamp_seconds, value). Out-of-window samples
        are dropped. Empty buckets render as blank cells (no zero bars).
    width: number of terminal cells. Each cell = 2 dot columns, so each
        bucket spans (t_end - t_start) / (2 * width) seconds.
    """
    if t_end <= t_start or width <= 0 or height <= 0:
        return [" " * width for _ in range(height)]
    n_cols = width * 2
    n_rows = height * 4
    span = t_end - t_start
    bucket_s = span / n_cols

    # Bucket samples by timestamp into n_cols slots.
    # bucket_vals[k] = list of values whose ts is in [t_start + k*bucket_s, ...)
    bucket_vals: list[list[float]] = [[] for _ in range(n_cols)]
    for ts, v in samples:
        if v is None or ts < t_start or ts > t_end:
            continue
        k = int((ts - t_start) / bucket_s)
        if k >= n_cols:
            k = n_cols - 1
        bucket_vals[k].append(v)

    # Mean per bucket; None for empty.
    bucket_means: list[float | None] = [
        (sum(vs) / len(vs)) if vs else None for vs in bucket_vals
    ]

    # Determine lo/hi from filled buckets (or use provided).
    filled = [m for m in bucket_means if m is not None]
    if not filled:
        return [" " * width for _ in range(height)]
    if lo is None:
        lo = min(filled)
    if hi is None:
        hi = max(filled)
    if hi <= lo:
        hi = lo + 1.0

    # For each bucket: level 0..n_rows. Empty buckets stay 0 and render blank.
    levels = [
        max(0, min(n_rows, round((m - lo) / (hi - lo) * n_rows)))
        if m is not None else None
        for m in bucket_means
    ]

    rows: list[str] = []
    for band in range(height):
        chars: list[str] = []
        for cell in range(width):
            bits = 0
            for col in range(2):
                lvl = levels[cell * 2 + col]
                if lvl is None:
                    continue
                for r_in_band in range(4):
                    abs_row = band * 4 + r_in_band
                    if (n_rows - abs_row) <= lvl:
                        bits |= _BRAILLE_BITS[(col, r_in_band)]
            chars.append(chr(0x2800 + bits))
        rows.append("".join(chars))
    return rows


def braille_bars(values, width: int, height: int,
                 lo: float | None = None, hi: float | None = None) -> list[str]:
    """Bottom-anchored Braille bar chart.

    width  cells  → 2*width data points
    height rows   → 4*height vertical levels
    Returns `height` strings, top row first.
    """
    n_cols = width * 2
    n_rows = height * 4
    vals = [v for v in values if v is not None][-n_cols:]
    if not vals:
        return [" " * width for _ in range(height)]
    if lo is None:
        lo = min(vals)
    if hi is None:
        hi = max(vals)
    if hi <= lo:
        hi = lo + 1.0
    levels = [max(0, min(n_rows, round((v - lo) / (hi - lo) * n_rows))) for v in vals]
    if len(levels) < n_cols:
        levels = [0] * (n_cols - len(levels)) + levels

    rows: list[str] = []
    for band in range(height):
        chars: list[str] = []
        for cell in range(width):
            bits = 0
            for col in range(2):
                lvl = levels[cell * 2 + col]
                for r_in_band in range(4):
                    abs_row = band * 4 + r_in_band
                    if (n_rows - abs_row) <= lvl:
                        bits |= _BRAILLE_BITS[(col, r_in_band)]
            chars.append(chr(0x2800 + bits))
        rows.append("".join(chars))
    return rows


_LINE_COLORS = (
    "bright_yellow", "bright_cyan", "bright_magenta",
    "bright_green", "bright_red", "bright_blue",
)


def line_chart(series: list[dict], width: int = 80, height: int = 18,
               *, x_label: str = "", y_label: str = "",
               title: str = "", x_log: bool = False,
               y_zero_base: bool = True) -> str:
    """Multi-series (x, y) line chart in Braille with axes + legend.

    Each series: ``{"label": str, "x": [...], "y": [...]}``.
    Returns a string with rich markup tags suitable for ``rich.console`` or
    Textual's ``Static``. Legend is rendered outside the plot area so it never
    obscures data points.

    y_zero_base: when True (default), force Y axis to start at zero so swings
        are proportional to absolute value. Disable for log-Y or for charts
        where the values are far from zero and zero-base wastes vertical space.
    """
    import math

    # Drop empty / mismatched series.
    series = [s for s in series if s.get("x") and s.get("y")
              and len(s["x"]) == len(s["y"])]
    if not series:
        return "[dim](no data)[/]"

    def x_map(v: float) -> float:
        return math.log10(v) if (x_log and v > 0) else v

    all_x = [x_map(x) for s in series for x in s["x"]]
    all_y = [y for s in series for y in s["y"]]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    if y_zero_base:
        y_min = min(0, y_min)  # force 0 baseline; negative values still shown
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1
    # 5% padding top so markers don't touch the frame. Skip bottom pad when
    # y_zero_base — we want the axis to literally sit at 0.
    y_pad = (y_max - y_min) * 0.05
    if not y_zero_base:
        y_min -= y_pad
    y_max += y_pad

    # Layout.
    y_label_w = 7  # 6 digits + separator
    title_rows = 1 if title else 0
    xaxis_rows = 2   # axis line + tick labels
    legend_rows = 1
    plot_rows = max(4, height - title_rows - xaxis_rows - legend_rows)
    plot_cols = max(20, width - y_label_w - 1)
    dot_cols = plot_cols * 2
    dot_rows = plot_rows * 4

    bits = [[0] * plot_cols for _ in range(plot_rows)]
    color = [[-1] * plot_cols for _ in range(plot_rows)]

    def to_dot(x: float, y: float) -> tuple[int, int]:
        dc = int(round((x_map(x) - x_min) / (x_max - x_min) * (dot_cols - 1)))
        dr = int(round((y_max - y) / (y_max - y_min) * (dot_rows - 1)))
        return dc, dr

    def set_dot(dc: int, dr: int, ci: int, mark: bool = False) -> None:
        if not (0 <= dc < dot_cols and 0 <= dr < dot_rows):
            return
        cc, cr = dc // 2, dr // 4
        if mark:
            bits[cr][cc] = 0xFF  # filled cell at the data point
        else:
            bits[cr][cc] |= _BRAILLE_BITS[(dc % 2, dr % 4)]
        color[cr][cc] = ci

    def bresenham(dc0: int, dr0: int, dc1: int, dr1: int):
        dx, dy = abs(dc1 - dc0), abs(dr1 - dr0)
        sx = 1 if dc0 < dc1 else -1
        sy = 1 if dr0 < dr1 else -1
        err = dx - dy
        while True:
            yield dc0, dr0
            if dc0 == dc1 and dr0 == dr1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                dc0 += sx
            if e2 < dx:
                err += dx
                dr0 += sy

    for i, s in enumerate(series):
        ci = i % len(_LINE_COLORS)
        pts = sorted(zip(s["x"], s["y"]), key=lambda p: p[0])
        dots = [to_dot(x, y) for x, y in pts]
        for (dc0, dr0), (dc1, dr1) in zip(dots, dots[1:]):
            for dc, dr in bresenham(dc0, dr0, dc1, dr1):
                set_dot(dc, dr, ci)
        for dc, dr in dots:
            set_dot(dc, dr, ci, mark=True)

    # ---- render ----------------------------------------------------------

    def fmt_y(v: float) -> str:
        if abs(v) >= 1000:
            return f"{v:6.0f}"
        if abs(v) >= 100:
            return f"{v:6.1f}"
        if abs(v) >= 10:
            return f"{v:6.2f}"
        return f"{v:6.3f}"

    out: list[str] = []
    if title:
        out.append(f"[bold]{title}[/]")

    # Y-axis tick rows: top, middle, bottom (and quartiles if there's room).
    label_rows: set[int] = {0, plot_rows - 1, plot_rows // 2}
    if plot_rows >= 9:
        label_rows |= {plot_rows // 4, (3 * plot_rows) // 4}
    for r in range(plot_rows):
        if r in label_rows:
            frac = r / max(1, plot_rows - 1)
            yv = y_max - frac * (y_max - y_min)
            ylab = fmt_y(yv)
        else:
            ylab = " " * (y_label_w - 1)
        # Build colored chunks across the row.
        chunks: list[str] = []
        cur_ci: int | None = None
        buf: list[str] = []
        for c in range(plot_cols):
            ch = chr(0x2800 + bits[r][c])
            ci = color[r][c]
            cname = _LINE_COLORS[ci] if ci >= 0 else None
            cur_name = _LINE_COLORS[cur_ci] if cur_ci is not None and cur_ci >= 0 else None
            if cname != cur_name:
                if buf:
                    if cur_name:
                        chunks.append(f"[{cur_name}]{''.join(buf)}[/]")
                    else:
                        chunks.append("".join(buf))
                buf = []
                cur_ci = ci
            buf.append(ch)
        if buf:
            if cur_ci is not None and cur_ci >= 0:
                chunks.append(f"[{_LINE_COLORS[cur_ci]}]{''.join(buf)}[/]")
            else:
                chunks.append("".join(buf))
        out.append(f"[dim]{ylab}│[/]" + "".join(chunks))

    # X-axis line.
    out.append(f"[dim]{' ' * (y_label_w - 1)}└" + "─" * plot_cols + "[/]")

    # X-axis labels at unique data x's. Anchor: left-edge for the leftmost
    # point, right-edge for the rightmost, centered otherwise — keeps labels
    # inside the plot area. Skip later labels that would collide.
    unique_x = sorted(set(x for s in series for x in s["x"]))
    # Generous buffer so right-anchored labels can't run past the edge.
    buf_len = y_label_w + plot_cols + 16
    xlab = [" "] * buf_len
    occupied = [False] * buf_len
    for i, x in enumerate(unique_x):
        col = int(round((x_map(x) - x_min) / (x_max - x_min) * (plot_cols - 1)))
        label = str(int(x)) if float(x).is_integer() else f"{x:g}"
        anchor_col = y_label_w + col
        if i == 0:
            pos = anchor_col  # left-anchor
        elif i == len(unique_x) - 1:
            pos = anchor_col - len(label) + 1  # right-anchor
        else:
            pos = anchor_col - len(label) // 2
        # Collision check: leave at least one space between labels.
        if pos > 0 and occupied[pos - 1]:
            continue
        clipped = False
        for j, ch in enumerate(label):
            p = pos + j
            if not (0 <= p < buf_len) or occupied[p]:
                clipped = True
                break
        if clipped:
            continue
        for j, ch in enumerate(label):
            p = pos + j
            xlab[p] = ch
            occupied[p] = True
    out.append("[dim]" + "".join(xlab).rstrip() + "[/]")

    # Legend (always below the plot — never overlaps data).
    legend_parts: list[str] = []
    for i, s in enumerate(series):
        cname = _LINE_COLORS[i % len(_LINE_COLORS)]
        legend_parts.append(f"[{cname}]●[/] {s['label']}")
    axis_meta = ""
    if x_label or y_label:
        axis_meta = f"  [dim](x={x_label}{', log' if x_log else ''}; y={y_label})[/]"
    out.append(" " * (y_label_w - 1) + "  ".join(legend_parts) + axis_meta)

    return "\n".join(out)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def classify_cell_regime(cell: dict, marginal_band: float = 0.10) -> dict:
    """Operator-actionable classification of a cell: does running this
    workload at concurrency actually gain throughput vs running it
    sequentially? Compares the cell's aggregate ceiling (from the
    hyperbolic fit, or the observed peak if no fit) against the c=1
    baseline.

    Returns: {
        "regime": "concurrency_wins" | "marginal" | "sequential_wins",
        "ratio": ceiling/c1_baseline,
        "label": short text for display (e.g. "+13% concurrent"),
        "color": rich style for display (green/yellow/red),
    }
    Or None if the cell has insufficient data to classify.

    marginal_band: how close to 1.0× counts as "doesn't matter which".
    Default 10%. Anything beyond that band is a clear directional call.

    Operator translation:
      concurrency_wins  — serve N users concurrently for more total t/s
      marginal          — within noise; either approach is fine
      sequential_wins   — concurrency hurts; queue and serve one at a
                          time, or use a different rig for high-c work
    """
    c1 = cell.get("c1_decode_tps")
    if c1 is None or c1 <= 0:
        return None
    fit = cell.get("fit")
    if fit and fit.get("aggregate_ceiling"):
        # Use the fit's ceiling — what aggregate would converge to at
        # high c if T_fixed amortized completely.
        ceiling = fit["aggregate_ceiling"]
    else:
        # Fall back to observed peak from the levels.
        peak_c = cell.get("aggregate_peak_c")
        ceiling = None
        if peak_c is not None:
            for lvl in cell.get("levels", []):
                if lvl["c"] == peak_c:
                    ceiling = lvl.get("aggregate_decode_tps")
                    break
        if ceiling is None:
            return None
    ratio = ceiling / c1
    if ratio > 1 + marginal_band:
        return {
            "regime": "concurrency_wins",
            "ratio": ratio,
            "label": f"+{(ratio - 1) * 100:.0f}% concurrent",
            "color": "green",
        }
    if ratio < 1 - marginal_band:
        return {
            "regime": "sequential_wins",
            "ratio": ratio,
            "label": f"−{(1 - ratio) * 100:.0f}% concurrent (serve sequentially)",
            "color": "red",
        }
    return {
        "regime": "marginal",
        "ratio": ratio,
        "label": f"{(ratio - 1) * 100:+.0f}% marginal",
        "color": "yellow",
    }


def intersect_run_cells(
    cells_per_run: list[dict],
) -> tuple[set, list[set]]:
    """Given a list of per-run cells dicts (each keyed by
    (preset, ctx, gen)), return:
      common_keys     — set of (preset, ctx, gen) present in ALL runs
      uniques_per_run — list of sets, one per run, of keys present only
                        in that run

    Used by weatherman A/B compare to honestly handle runs with
    different sweep matrices (e.g., older runs from before a ctx
    sweep was revised). The intersection is what's directly
    comparable; the uniques footer tells the operator what's missing
    from where.
    """
    if not cells_per_run:
        return set(), []
    keys_per_run = [set(c.keys()) for c in cells_per_run]
    common = set.intersection(*keys_per_run) if keys_per_run else set()
    uniques = [k - common for k in keys_per_run]
    return common, uniques


def interpolate_cell_metrics_at_ctx(
    cells_dict: dict,
    preset: str, target_ctx: int, gen: int,
) -> dict | None:
    """Linearly interpolate cell metrics at target_ctx using the
    bracketing measured ctx values for the same (preset, gen). Returns
    None if:
      - target_ctx is outside the measured range (no extrapolation —
        operator-honesty boundary)
      - fewer than 2 measured points exist for that (preset, gen)
      - target_ctx exactly matches a measured ctx (caller should use
        the measured cell directly instead of interpolating)

    Interpolated fields:
      c1_decode_tps, aggregate_peak_value, fit.t_fixed,
      fit.t_per_stream, fit.aggregate_ceiling, fit.c1_baseline

    Returns a dict with `interpolated: True`, `ctx_lower` / `ctx_upper`
    showing which measured points bracket the projection — so callers
    can flag the result visually (◇ vs ●).
    """
    measured = sorted([
        (cs, cell) for (p, cs, gs), cell in cells_dict.items()
        if p == preset and gs == gen
    ])
    if len(measured) < 2:
        return None
    for cs, _ in measured:
        if cs == target_ctx:
            return None
    lower = None
    upper = None
    for cs, cell in measured:
        if cs < target_ctx:
            lower = (cs, cell)
        elif cs > target_ctx and upper is None:
            upper = (cs, cell)
            break
    if lower is None or upper is None:
        return None
    cs_lo, cell_lo = lower
    cs_hi, cell_hi = upper
    frac = (target_ctx - cs_lo) / (cs_hi - cs_lo)

    def lerp(a, b):
        if a is None or b is None:
            return None
        return a + (b - a) * frac

    result: dict = {
        "interpolated": True,
        "ctx": target_ctx,
        "ctx_lower": cs_lo,
        "ctx_upper": cs_hi,
        "c1_decode_tps": lerp(cell_lo.get("c1_decode_tps"),
                              cell_hi.get("c1_decode_tps")),
        "aggregate_peak_c": cell_lo.get("aggregate_peak_c"),  # discrete; carry lower's
    }
    # Aggregate peak value at the lower's peak_c, interpolated.
    def peak_value(cell):
        pc = cell.get("aggregate_peak_c")
        if pc is None:
            return None
        for lvl in cell.get("levels", []):
            if lvl["c"] == pc:
                return lvl.get("aggregate_decode_tps")
        return None
    result["aggregate_peak_value"] = lerp(peak_value(cell_lo), peak_value(cell_hi))
    fit_lo = cell_lo.get("fit")
    fit_hi = cell_hi.get("fit")
    if fit_lo and fit_hi:
        result["fit"] = {
            "t_fixed": lerp(fit_lo.get("t_fixed"), fit_hi.get("t_fixed")),
            "t_per_stream": lerp(fit_lo.get("t_per_stream"),
                                 fit_hi.get("t_per_stream")),
            "aggregate_ceiling": lerp(fit_lo.get("aggregate_ceiling"),
                                      fit_hi.get("aggregate_ceiling")),
            "c1_baseline": lerp(fit_lo.get("c1_baseline"),
                                fit_hi.get("c1_baseline")),
        }
    return result


def narrate_run_events(run_dir: Path) -> str:
    """Convert a run's events.log into a markdown narrative — readable
    prose summarizing what kata did during the run, suitable for
    embedding in blog posts, video scripts, or shared write-ups.

    Groups events by cell (each cell_start opens a section), then within
    each cell renders level completions + knee labels + termination
    chronologically. Returns the full markdown string. Returns an empty
    string if the run has no events.log.
    """
    events = load_events(run_dir)
    if not events:
        return ""

    sys = load_system(run_dir)
    rig = sys.get("rig_label", "?")
    cfg = sys.get("config") or {}
    name = cfg.get("name", "(unnamed)")
    host = (sys.get("host") or {}).get("hostname", "?")

    # Header.
    first_ts = events[0].get("ts_local", "")
    last_ts = events[-1].get("ts_local", "")
    parts: list[str] = [
        f"# Narrative: {name}",
        "",
        f"**Rig:** {rig} (`{host}`)  ",
        f"**Window:** {first_ts} → {last_ts}  ",
        f"**Events:** {len(events)}",
        "",
    ]

    # Walk events grouped by cell. Each cell_start opens a section;
    # everything until the next cell_start or end belongs to that cell.
    current_section: list[dict] | None = None
    sections: list[list[dict]] = []
    preamble: list[dict] = []
    for ev in events:
        if ev.get("type") == "cell_start":
            if current_section is not None:
                sections.append(current_section)
            current_section = [ev]
        elif current_section is not None:
            current_section.append(ev)
        else:
            preamble.append(ev)
    if current_section is not None:
        sections.append(current_section)

    if preamble:
        parts.append("## Setup")
        parts.append("")
        for ev in preamble:
            parts.append(f"- `{ev.get('ts_local', '')}` {ev.get('msg', '')}")
        parts.append("")

    for section in sections:
        first = section[0]
        preset = first.get("preset", "?")
        ctx = first.get("context_size", "?")
        gen = first.get("gen_size", "?")
        parts.append(f"## `{preset}` ctx={ctx} gen={gen}")
        parts.append("")
        parts.append(f"Started at {first.get('ts_local', '?')}.")
        parts.append("")
        baseline_msg = None
        levels = []
        knees = []
        termination = None
        for ev in section[1:]:
            t = ev.get("type")
            if t == "baseline_complete":
                baseline_msg = ev
            elif t == "level_complete":
                levels.append(ev)
            elif t in ("knee_per_req", "saturation_label"):
                knees.append(ev)
            elif t == "termination":
                termination = ev
        if baseline_msg:
            parts.append(f"- c=1 baseline: **{baseline_msg.get('per_req_decode_tps', 0):.1f} t/s** "
                         f"(T_req={baseline_msg.get('t_req_s', 0):.2f}s, "
                         f"{baseline_msg.get('n_steady', 0)} steady samples)")
        for ev in levels:
            c = ev.get("c")
            per_req = ev.get("per_req_decode_tps") or 0
            agg = ev.get("aggregate_decode_tps") or 0
            mode = ev.get("mode", "")
            frac = (ev.get("per_req_frac_of_c1") or 0) * 100
            parts.append(f"- c={c} ({mode}): per-req {per_req:.1f} t/s "
                         f"({frac:.0f}% of c=1), aggregate {agg:.1f} t/s")
        for ev in knees:
            parts.append(f"- `{ev.get('ts_local', '')}` {ev.get('msg', '')}")
        if termination:
            parts.append(f"- **Termination:** {termination.get('msg', '')}")
        parts.append("")

    return "\n".join(parts)


def search_events_across_runs(runs_dir: Path, query: str,
                              limit: int = 200) -> list[dict]:
    """Walk all events.log files under runs_dir, return events whose msg
    (or any structured field value) contains the query string (case
    insensitive). Each result includes `run_name` so the operator can
    navigate back to the source run.

    Used by weatherman's search panel — answers "find all runs that
    mentioned 'Host memory'" or "show me every aggregate_plateau
    termination across the tree" without manually drilling into each
    run's Events tab.

    Result limit prevents huge result sets from hanging the UI.
    """
    if not query:
        return []
    q = query.lower()
    results: list[dict] = []
    if not runs_dir.exists():
        return results
    for run_path in sorted(runs_dir.iterdir(), reverse=True):
        if not run_path.is_dir():
            continue
        events = load_events(run_path)
        if not events:
            continue
        for ev in events:
            # Build a haystack of stringified values from msg + structured
            # fields. Skip meta fields used for indexing.
            hay_parts = [str(ev.get("msg", ""))]
            for k, v in ev.items():
                if k in ("ts", "ts_local", "ts_unix"):
                    continue
                hay_parts.append(f"{k}={v}")
            hay = " ".join(hay_parts).lower()
            if q in hay:
                results.append({**ev, "run_name": run_path.name})
                if len(results) >= limit:
                    return results
    return results


def load_events(run_dir: Path) -> list[dict]:
    """Load events.log JSONL for a run, in chronological order. Returns
    empty list if no events.log exists (e.g. older runs predating the
    event-log feature). Each event has at minimum ts_local, level, msg;
    type and structured fields when present."""
    return load_jsonl(run_dir / "events.log")


def run_has_warnings(run_dir: Path) -> bool:
    """Quick check: did this run record any warn/error events? Used by
    weatherman to mark suspect runs in the tree without loading all
    events. Reads only the events.log file; cheap O(file size)."""
    for ev in load_events(run_dir):
        if ev.get("level") in ("warn", "error"):
            return True
    return False


def load_system(run_dir: Path) -> dict:
    p = run_dir / "system.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def load_summary(run_dir: Path) -> dict:
    p = run_dir / "summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


# Quant suffix regex: matches the usual GGUF tags (Q4_K_M, IQ3_XXS, F16, BF16, Q8_0, ...)
_QUANT_RE = re.compile(
    r"(?P<q>(?:IQ|Q)\d(?:_[A-Z0-9]+)*|F16|F32|BF16)",
    re.IGNORECASE,
)


def parse_model_quant(server_args: dict) -> tuple[str, str]:
    """
    Extract a (model, quant) label from launch args.
    Looks at --hf-repo / -hf / --model first, falls back to "unknown".
    """
    for key in ("--hf-repo", "-hf", "--model", "-m"):
        if key in server_args and server_args[key]:
            raw = str(server_args[key])
            # split on colon (hf form is repo:quant) or path
            base = raw.split("/")[-1]
            name, _, tag = base.partition(":")
            quant_src = tag or name
            m = _QUANT_RE.search(quant_src)
            quant = m.group("q").upper() if m else "?"
            # strip the quant fragment from the model name for cleaner display
            clean = _QUANT_RE.sub("", name).strip(" -_.")
            # drop trailing "-GGUF" decoration and trailing ".gguf"
            clean = re.sub(r"[-_.]GGUF$", "", clean, flags=re.IGNORECASE)
            clean = re.sub(r"\.gguf$", "", clean, flags=re.IGNORECASE).strip(" -_.")
            return clean or name, quant
    return "unknown", "?"


def server_args_to_dict(args: list) -> dict:
    """
    Convert YAML server.args (list of mappings or strings) into a flat dict.
    YAML form:  - --hf-repo: foo/bar   →   {"--hf-repo": "foo/bar"}
                - -fa                  →   {"-fa": True}
    """
    out: dict[str, Any] = {}
    for entry in args or []:
        if isinstance(entry, dict):
            for k, v in entry.items():
                out[str(k)] = v
        elif isinstance(entry, str):
            out[entry] = True
    return out


def server_args_to_argv(args: list) -> list[str]:
    """Convert YAML server.args into a flat argv list for subprocess."""
    argv: list[str] = []
    for entry in args or []:
        if isinstance(entry, dict):
            for k, v in entry.items():
                argv.append(str(k))
                if v is True or v is None:
                    continue
                argv.append(str(v))
        elif isinstance(entry, str):
            argv.append(entry)
    return argv


# Server args we don't show in the parameter summary (redundant / boilerplate).
_PARAM_HIDE = {
    "--hf-repo", "-hf", "--model", "-m",   # model id (shown separately)
    "--model-draft", "-md",                 # draft model (shown separately)
    "--host", "--port",                     # network plumbing
    "--spec-type",                          # shown in dedicated spec line
}

# Categorize server args so the summary panel groups them visually.
# Order matters — earlier groups render first.
_PARAM_GROUPS: list[tuple[str, set[str]]] = [
    ("compute", {
        "--batch-size", "-b", "--ubatch-size", "-ub", "--parallel", "-np",
        "--ctx-size", "-c", "-fa", "-ngl", "--n-gpu-layers",
        "--threads", "-t", "--threads-batch",
        "--tensor-split", "--main-gpu", "--split-mode",
    }),
    ("cache", {
        "--cache-type-k", "-ctk", "--cache-type-v", "-ctv",
        "--kv_unified", "--no-kv-offload", "--no-context-shift",
    }),
    ("spec", {
        "--spec-draft-n-max", "--spec-draft-n-min", "--spec-p-min",
        "--spec-ngram-mod-n-match", "--spec-ngram-mod-n-min",
        "--spec-ngram-mod-n-max",
        "--draft-max", "--draft-min", "--draft-p-min",
    }),
    ("misc", {
        "--jinja", "--reasoning", "--fit",
    }),
]


def categorize_server_args(server_args: dict) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str]]]:
    """Return (grouped, ungrouped). `grouped[name]` = list of (flag, value-str).

    Hidden flags are dropped; unrecognized flags land in `ungrouped`.
    """
    grouped: dict[str, list[tuple[str, str]]] = {name: [] for name, _ in _PARAM_GROUPS}
    ungrouped: list[tuple[str, str]] = []
    flag_to_group = {f: name for name, flags in _PARAM_GROUPS for f in flags}
    for flag, val in server_args.items():
        if flag in _PARAM_HIDE:
            continue
        s = "" if val is True else str(val)
        group = flag_to_group.get(flag)
        entry = (flag.lstrip("-"), s)
        if group:
            grouped[group].append(entry)
        else:
            ungrouped.append(entry)
    return grouped, ungrouped


def index_run(run_dir: Path) -> RunIndex | None:
    """Build a RunIndex from a single run directory. Returns None if unusable."""
    system = load_system(run_dir)
    summary = load_summary(run_dir)
    rows = load_jsonl(run_dir / "raw.jsonl")

    if not system and not summary and not rows:
        return None

    config = system.get("config", {})
    server_args = config.get("server_args", {})
    model, quant = parse_model_quant(server_args)
    draft_model = server_args.get("--model-draft") or server_args.get("-md")

    gpus = system.get("gpus", []) or []
    selection = (system.get("selection") or {}).get("cuda_visible_devices")
    if selection:
        picks = [s.strip() for s in str(selection).split(",") if s.strip() != ""]
        selected = [g for g in gpus if str(g.get("index")) in picks]
    else:
        selected = gpus
    if selected:
        names = [g.get("name", "?") for g in selected]
        if len(set(names)) == 1:
            gpus_selected = f"{len(names)}x {names[0]}" if len(names) > 1 else names[0]
        else:
            gpus_selected = ", ".join(names)
    else:
        gpus_selected = "cpu/unknown"

    # Token-weighted decode / prefill means computed from raw rows.
    decode_mean = prefill_mean = accept_mean = ttft_mean = None
    if rows:
        pred_t = [(r["predicted_n"] / r["decode_tps"]) for r in rows
                  if (r.get("decode_tps") or 0) > 0 and r.get("predicted_n")]
        pred_n = [r.get("predicted_n") or 0 for r in rows
                  if (r.get("decode_tps") or 0) > 0]
        if pred_t and sum(pred_t) > 0:
            decode_mean = sum(pred_n) / sum(pred_t)

        prompt_t = [(r["prompt_n"] / r["prefill_tps"]) for r in rows
                    if (r.get("prefill_tps") or 0) > 0 and r.get("prompt_n")]
        prompt_n = [r.get("prompt_n") or 0 for r in rows
                    if (r.get("prefill_tps") or 0) > 0]
        if prompt_t and sum(prompt_t) > 0:
            prefill_mean = sum(prompt_n) / sum(prompt_t)

        accept_vals = [
            (r["draft_accepted"] / r["draft_n"])
            for r in rows
            if r.get("draft_n") and r.get("draft_accepted") is not None
        ]
        accept_mean = mean(accept_vals) if accept_vals else None

        ttft_vals = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
        ttft_mean = mean(ttft_vals) if ttft_vals else None

    raw_spec = server_args.get("--spec-type")
    spec_type = str(raw_spec) if raw_spec and str(raw_spec).lower() != "none" else None

    # Presets present in the run (sorted, unique).
    presets = sorted({r.get("preset") for r in rows if r.get("preset")})

    return RunIndex(
        path=run_dir,
        timestamp=system.get("timestamp_utc") or run_dir.name,
        rig_label=system.get("rig_label", "?"),
        model=model,
        quant=quant,
        draft_model=str(draft_model) if draft_model else None,
        spec_type=spec_type,
        presets=presets,
        gpus_selected=gpus_selected,
        commit=(system.get("llama_cpp") or {}).get("git_commit"),
        decode_tps_mean=decode_mean,
        prefill_tps_mean=prefill_mean,
        ttft_ms_mean=ttft_mean,
        accept_rate_mean=accept_mean,
        rows=len(rows),
    )


def discover_runs(runs_dir: Path) -> list[RunIndex]:
    """Scan a runs directory for valid run subdirs."""
    if not runs_dir.exists():
        return []
    out: list[RunIndex] = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        idx = index_run(child)
        if idx is not None:
            out.append(idx)
    return out


# ---------- timestamp display --------------------------------------------
# system.json stores `timestamp_utc` for portability across machines, but
# UI sites should show the local time so it matches what the user saw on
# the wall clock when the run happened.


def format_timestamp_local(ts_str: str | None, *, with_tz: bool = True) -> str:
    """Convert an ISO timestamp string (UTC or naive) to local-time display.

    Accepts the formats we actually emit: `2026-06-03T13:34:38.408781+00:00`
    and fallback to the run-dir name if parsing fails.
    """
    if not ts_str:
        return "?"
    import datetime
    try:
        # Python's fromisoformat accepts +00:00 and microseconds directly.
        dt = datetime.datetime.fromisoformat(ts_str)
    except ValueError:
        return ts_str  # not a parseable timestamp — pass through unchanged
    if dt.tzinfo is None:
        # Naive timestamp: assume UTC (that's what we emit historically).
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    local = dt.astimezone()  # converts to system local tz
    if with_tz:
        return local.strftime("%Y-%m-%d %H:%M:%S %Z")
    return local.strftime("%Y-%m-%d %H:%M:%S")


# ---------- prompt presets -----------------------------------------------
# Each preset wraps a corpus + a trailing "task" that elicits a particular
# output domain. The output domain matters a lot for speculative-decoding
# acceptance: code → high accept, prose → mid, retrieval → mid. Presets are
# the lever to expose that effect across a single sweep.

PROMPT_PRESETS: dict[str, dict] = {
    "code": {
        "description": "Python source corpus + refactor request. High spec-accept domain.",
        "corpus_path": "prompts/code_corpus.py",
        "task_template": (
            "\n\n# ===== TASK =====\n"
            "# Refactor the code above. Improve naming, add type hints where\n"
            "# missing, extract helpers where readability benefits, and add a\n"
            "# brief one-line docstring to each public function. Preserve all\n"
            "# behavior exactly. Output only Python code, no commentary.\n"
            "\n"
        ),
    },
    "chat": {
        "description": "Multi-turn dialogue + summarization task. Conversational decode.",
        "corpus_path": "prompts/chat_corpus.txt",
        "task_template": (
            "\n\n"
            "User: Summarize the conversation(s) above in 5 concise bullet\n"
            "points, then suggest one follow-up question the user might ask\n"
            "next.\n"
            "\n"
            "Assistant:"
        ),
    },
    "niah": {
        "description": ("Pride & Prejudice — needle-in-a-haystack with retrieval "
                        "questions targeting an always-present middle passage."),
        "corpus_path": "prompts/niah_corpus.txt",
        # The anchor passage must always be present in the prompt — it contains
        # the answers to the canned questions. Identified by literal string
        # markers (start, end) that occur exactly once in the corpus.
        "anchor_start_marker": "from Rosings, at eight o",
        "anchor_end_marker":   "He spoke of it as a certain",
        "questions": [
            "At what time and from where is Mr. Darcy's letter to Elizabeth dated? "
                "Quote the exact phrase.",
            "What two offences did Elizabeth charge Mr. Darcy with on the previous "
                "night? Name both, in the order Mr. Darcy lists them.",
            "Who first informed Mr. Darcy that Bingley's attentions to Elizabeth's "
                "sister had given rise to a general expectation of their marriage? "
                "Quote the source.",
        ],
        "task_template_fmt": (
            "\n\n"
            "User: Based only on the passage above, answer this question concisely. "
            "Quote the relevant sentence(s) from the text in your answer.\n\n"
            "Question: {question}\n\n"
            "Assistant:"
        ),
    },
}


def list_prompt_presets() -> list[str]:
    return list(PROMPT_PRESETS.keys())


def _resolve_repo_path(rel: str) -> Path:
    """Resolve a path relative to the katabasis repo root (where bench_data.py lives)."""
    here = Path(__file__).resolve().parent
    return here / rel


def _load_preset_corpus(preset_name: str) -> str:
    """Load the raw corpus text for a preset."""
    p = PROMPT_PRESETS.get(preset_name)
    if p is None:
        raise KeyError(f"unknown prompt preset: {preset_name!r}. "
                       f"Available: {list_prompt_presets()}")
    path = _resolve_repo_path(p["corpus_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"preset {preset_name!r} expected corpus at {path}, but it's missing. "
            "Bundled corpora live in prompts/ — re-clone or restore the file."
        )
    return path.read_text()


def normalize_preset_spec(spec) -> tuple[str, str | None]:
    """Accept a string preset name or {file: path} dict; return (name, path|None).

    The 'name' is what gets recorded in raw.jsonl rows. For file-based custom
    corpora the name is 'file:<basename>' so it's still grouped sensibly in
    weatherman.
    """
    if isinstance(spec, str):
        return spec, None
    if isinstance(spec, dict) and "file" in spec:
        path = str(spec["file"])
        return f"file:{Path(path).name}", path
    raise ValueError(f"invalid prompt preset spec: {spec!r}")


def build_prompt(spec, target_tokens: int, *,
                 tokenize_fn, detokenize_fn) -> tuple[str, str]:
    """Build a prompt of `target_tokens` tokens for a given preset spec.

    Args:
        spec: preset name (str) or {file: path} (dict).
        target_tokens: total prompt length in tokens, including the task footer.
        tokenize_fn(text) -> list[int]: round-trips text through the server's
            /tokenize endpoint so token counts match exactly what the model sees.
        detokenize_fn(tokens) -> str: inverse of tokenize_fn.

    Returns:
        (preset_name, prompt_text)
    """
    name, file_path = normalize_preset_spec(spec)

    # Resolve corpus + task. NIAH has a custom build path (anchor-preserving).
    if file_path is not None:
        corpus_text = Path(file_path).read_text()
        task_template = ""
        preset = None
    else:
        preset = PROMPT_PRESETS.get(name)
        if preset is None:
            raise KeyError(
                f"unknown preset: {name!r}. Available: {list_prompt_presets()}"
            )
        corpus_text = _load_preset_corpus(name)
        task_template = preset.get("task_template", "")

    # --- NIAH: anchor-preserving build path ---
    if preset is not None and "anchor_start_marker" in preset:
        return _build_niah_prompt(
            name, preset, corpus_text, target_tokens,
            tokenize_fn=tokenize_fn, detokenize_fn=detokenize_fn,
        )

    # --- Standard: truncate + append task ---
    task_tokens = tokenize_fn(task_template) if task_template else []
    task_n = len(task_tokens)

    if target_tokens <= task_n:
        return name, detokenize_fn(task_tokens[:max(1, target_tokens)])

    body_target = target_tokens - task_n
    tokens: list[int] = tokenize_fn(corpus_text)
    if len(tokens) >= body_target:
        body_tokens = tokens[:body_target]
    else:
        sep = "\n\n# --- corpus continues (recycled) ---\n\n"
        sep_tokens = tokenize_fn(sep)
        chunks: list[int] = []
        while len(chunks) < body_target:
            if chunks:
                chunks.extend(sep_tokens)
                if len(chunks) >= body_target:
                    break
            chunks.extend(tokens)
        body_tokens = chunks[:body_target]

    return name, detokenize_fn(body_tokens) + task_template


def _build_niah_prompt(name: str, preset: dict, corpus_text: str,
                       target_tokens: int, *,
                       tokenize_fn, detokenize_fn) -> tuple[str, str]:
    """NIAH: locate the anchor passage in the corpus, pad symmetrically around
    it, and append a rotated question. Anchor is never truncated — if the
    target is too small to fit anchor+task, we exceed it (and warn-by-shape)."""
    start_marker = preset["anchor_start_marker"]
    end_marker = preset["anchor_end_marker"]
    questions = preset["questions"]
    task_fmt = preset["task_template_fmt"]

    # Pick question deterministically — different ctx sizes ask different
    # questions naturally, while a given ctx is stable across rounds. Plain
    # modulo collides on powers-of-2 contexts (typical sweep values); adding
    # bit_length spreads consecutive powers of 2 across the question set.
    q_idx = (target_tokens.bit_length() + target_tokens) % len(questions)
    task_template = task_fmt.format(question=questions[q_idx])
    task_tokens = tokenize_fn(task_template)
    task_n = len(task_tokens)

    # Locate the anchor passage in the corpus. Anchor spans from start_marker
    # through end_marker INCLUDING the end marker's text — we want full
    # sentences on both sides of the answer-bearing facts.
    start_idx = corpus_text.find(start_marker)
    end_idx = corpus_text.find(end_marker, start_idx + 1) if start_idx >= 0 else -1
    if start_idx < 0 or end_idx < 0:
        # Corpus doesn't contain the expected anchor markers — degrade to
        # full-corpus truncation (still better than nothing).
        return name, (corpus_text[:max(0, target_tokens * 5)] + task_template)
    anchor_end_pos = end_idx + len(end_marker)

    pre_anchor_text = corpus_text[:start_idx]
    anchor_text = corpus_text[start_idx:anchor_end_pos]
    post_anchor_text = corpus_text[anchor_end_pos:]

    anchor_tokens = tokenize_fn(anchor_text)
    anchor_n = len(anchor_tokens)

    # If anchor + task already exceeds target, return anchor + task as-is.
    if anchor_n + task_n >= target_tokens:
        return name, anchor_text + task_template

    # Otherwise pad symmetrically with surrounding book text.
    pad_budget = target_tokens - anchor_n - task_n
    pad_each = pad_budget // 2
    extra = pad_budget - 2 * pad_each  # 0 or 1 leftover token — give to post

    pre_tokens = tokenize_fn(pre_anchor_text)
    post_tokens = tokenize_fn(post_anchor_text)

    # Take the last `pad_each` tokens of pre-anchor (closest to anchor),
    # and the first `pad_each + extra` tokens of post-anchor.
    pre_keep = pre_tokens[-pad_each:] if pad_each > 0 else []
    post_keep = post_tokens[:pad_each + extra] if (pad_each + extra) > 0 else []

    body_tokens = pre_keep + anchor_tokens + post_keep
    body_text = detokenize_fn(body_tokens)
    return name, body_text + task_template


def get_context_sizes(sweep_cfg: dict) -> list[int]:
    return list(sweep_cfg.get("context_sizes") or [])


def get_prompt_presets(sweep_cfg: dict) -> list[str | dict]:
    """Read the prompt_presets list from a sweep config (always a list)."""
    return list(sweep_cfg.get("prompt_presets") or [])


def get_max_concurrency(sweep_cfg: dict) -> int:
    """Read max_concurrency — the ramp's safety cap. The rolling driver may
    abort earlier when the per-request decode_tps drops below the configured
    fraction of the c=1 baseline (the knee). Default 16."""
    return int(sweep_cfg.get("max_concurrency") or 16)


def fit_ramp_per_req(measured: list[tuple[int, float]]) -> dict | None:
    """Fit a physically-motivated per-request throughput model from
    (c, per_req_decode_tps) pairs. Model:

        per_req(c) = 1 / (T_fixed + T_per_stream * c)        for c >= 2

    Physical reading: each decode step pays a fixed batch overhead
    (kernel launches, slot management, attention setup) plus per-stream
    cost (the actual decode compute / memory bandwidth work for one
    token of one stream). As c grows, T_fixed amortizes across more
    streams and aggregate throughput climbs toward 1/T_per_stream as
    an asymptote.

    Fit via least-squares linear regression of 1/per_req vs c on c>=2
    samples (c=1 is a separate code path — single-stream optimizations
    avoid most of T_fixed — so it's recorded as c1_baseline but excluded
    from the fit). Needs at least 2 distinct c>=2 points to fit a line.

    Returns: {
      t_fixed,           # batch-overhead time per decode step (s/token)
      t_per_stream,      # per-stream marginal time (s/token/stream)
      c1_baseline,       # measured per_req at c=1 (separate code path)
      aggregate_ceiling, # 1/t_per_stream — asymptote of aggregate t/s
    } or None when there isn't enough data to fit.
    """
    if not measured:
        return None
    measured = sorted(measured, key=lambda x: x[0])
    c1_baseline = next((v for c, v in measured if c == 1), None)
    fit_pts = [(c, 1.0 / v) for c, v in measured if c >= 2 and v > 0]
    if len(fit_pts) < 2:
        return None
    n = len(fit_pts)
    sx = sum(c for c, _ in fit_pts)
    sy = sum(y for _, y in fit_pts)
    sxx = sum(c * c for c, _ in fit_pts)
    sxy = sum(c * y for c, y in fit_pts)
    denom = n * sxx - sx * sx
    if denom <= 0:
        return None
    b = (n * sxy - sx * sy) / denom   # T_per_stream
    a = (sy - b * sx) / n             # T_fixed (unconstrained)

    # T_fixed must be physically non-negative — it represents fixed per-batch
    # overhead time. A negative unconstrained intercept means the data
    # doesn't actually have an overhead-amortization shape (typical for
    # systems already saturated at c=1, where aggregate doesn't climb with c).
    # Constrain by forcing T_fixed = 0 and refitting T_per_stream alone:
    # minimize sum((y_i - b*c_i)^2) → b = sum(c_i*y_i) / sum(c_i^2).
    # That reduces the model to pure c1/c (per_req = 1/(T_per_stream * c)),
    # which is the right answer for compute-saturated systems.
    if a < 0:
        a = 0.0
        b = sxy / sxx if sxx > 0 else None
    if b is None or b <= 0:
        return None  # slope must be positive for per_req to be finite and >0
    return {
        "t_fixed": a,
        "t_per_stream": b,
        "c1_baseline": c1_baseline,
        "aggregate_ceiling": 1.0 / b,
    }


def predict_ramp_per_req(c: int, fit: dict | None) -> float | None:
    """Predict per-req throughput at concurrency c under the fitted model.

    Returns the model's prediction even at c=1 — the gap between the
    fit's c=1 prediction and the measured c=1 baseline is informative
    (it quantifies how much T_fixed the single-stream code path avoids).
    Callers that want to draw c=1 from measurement should use
    fit["c1_baseline"] explicitly."""
    if fit is None:
        return None
    denom = fit["t_fixed"] + fit["t_per_stream"] * c
    if denom <= 0:
        return None
    return 1.0 / denom


def aggregate_ramp_by_cell(rows: list[dict]) -> dict[tuple[str, int, int], dict]:
    """Group rolling-ramp rows by (preset, context_size, gen_size) and
    summarize each cell's ramp shape using steady-state samples only.

    Returns: {(preset, ctx, gen): {
        c1_decode_tps,
        fit,                          # hyperbolic-fit dict (T_fixed,
                                      # T_per_stream, c1_baseline,
                                      # aggregate_ceiling) or None
        aggregate_peak_c,             # argmax aggregate over measured levels
        per_req_threshold_c,          # first c where per_req < 0.5 × c1
        levels: [{
            c, n_steady, n_total,
            per_req_decode_tps,       # token-weighted across steady samples
            predicted_per_req,        # from hyperbolic fit
            deviation,                # measured/predicted - 1
            aggregate_decode_tps,
            wall_mean_s,
            ttft_ms_mean, ttft_ms_p95,
            prefill_tps_tw,
            accept_mean,
        }, ...]
    }}
    Note: stop_reason and saturated_from_c are NOT carried in this
    aggregate — those come from kata.py:run_ramp_cell's return dict
    and aren't reconstructable from raw.jsonl alone. Consumers needing
    those should pull from events.log instead.
    """
    from collections import defaultdict
    cells: dict[tuple[str, int, int], dict[int, dict]] = defaultdict(lambda: defaultdict(list))
    # First pass: bucket rows by (preset, ctx, gen, c).
    for r in rows:
        key = (r.get("preset"), r.get("context_size"), r.get("gen_size"))
        if None in key:
            continue
        c = r.get("concurrency")
        if c is None:
            continue
        cells[key].setdefault(c, []).append(r)

    out: dict[tuple[str, int, int], dict] = {}
    for key, by_c in cells.items():
        levels: list[dict] = []
        c1_decode_tps: float | None = None
        # Sort c levels ascending so c=1 is first.
        for c in sorted(by_c.keys()):
            all_rows = by_c[c]
            steady_rows = [r for r in all_rows if r.get("in_steady_state")]
            # Token-weighted per-req decode rate from steady samples.
            pairs = [(r["predicted_n"], r["predicted_n"] / r["decode_tps"])
                     for r in steady_rows
                     if (r.get("decode_tps") or 0) > 0 and r.get("predicted_n")]
            if pairs:
                total_tok = sum(p[0] for p in pairs)
                total_t = sum(p[1] for p in pairs)
                per_req_tw = (total_tok / total_t) if total_t > 0 else None
            else:
                per_req_tw = None

            pre_pairs = [(r["prompt_n"], r["prompt_n"] / r["prefill_tps"])
                         for r in steady_rows
                         if (r.get("prefill_tps") or 0) > 0 and r.get("prompt_n")]
            if pre_pairs:
                prefill_tw = sum(p[0] for p in pre_pairs) / sum(p[1] for p in pre_pairs)
            else:
                prefill_tw = None

            ttft_vals = sorted(r["ttft_ms"] for r in steady_rows if r.get("ttft_ms") is not None)
            ttft_mean_v = (sum(ttft_vals) / len(ttft_vals)) if ttft_vals else None
            ttft_p95 = ttft_vals[int(0.95 * (len(ttft_vals) - 1))] if ttft_vals else None
            wall_vals = [r["wall_s"] for r in steady_rows if r.get("wall_s") is not None]
            wall_mean = (sum(wall_vals) / len(wall_vals)) if wall_vals else None
            accept_vals = [r["draft_accepted"] / r["draft_n"]
                           for r in steady_rows
                           if r.get("draft_n") and r.get("draft_accepted") is not None]
            accept_mean_v = (sum(accept_vals) / len(accept_vals)) if accept_vals else None

            if c == 1 and per_req_tw is not None:
                c1_decode_tps = per_req_tw

            aggregate = (per_req_tw * c) if per_req_tw is not None else None
            levels.append({
                "c": c,
                "n_steady": len(steady_rows),
                "n_total": len(all_rows),
                "per_req_decode_tps": per_req_tw,
                "aggregate_decode_tps": aggregate,
                "wall_mean_s": wall_mean,
                "ttft_ms_mean": ttft_mean_v,
                "ttft_ms_p95": ttft_p95,
                "prefill_tps_tw": prefill_tw,
                "accept_mean": accept_mean_v,
            })

        # Fit per_req(c) from measured per-req values (steady-state only).
        # New model: 1/per_req = T_fixed + T_per_stream * c (fit on c>=2).
        measured_for_fit = [(lvl["c"], lvl["per_req_decode_tps"])
                            for lvl in levels if lvl["per_req_decode_tps"] is not None]
        fit = fit_ramp_per_req(measured_for_fit)
        for lvl in levels:
            if fit is None or lvl["per_req_decode_tps"] is None:
                lvl["predicted_per_req"] = None
                lvl["deviation"] = None
                continue
            pred = predict_ramp_per_req(lvl["c"], fit)
            lvl["predicted_per_req"] = pred
            lvl["deviation"] = (lvl["per_req_decode_tps"] / pred - 1) if pred and pred > 0 else None

        # Argmax aggregate.
        valid_agg = [lvl for lvl in levels if lvl["aggregate_decode_tps"] is not None]
        agg_peak_c = (max(valid_agg, key=lambda l: l["aggregate_decode_tps"])["c"]
                      if valid_agg else None)
        # First c where per_req drops below 50% of c=1.
        threshold_c = None
        if c1_decode_tps:
            for lvl in levels:
                if lvl["per_req_decode_tps"] is not None and lvl["c"] > 1 \
                        and lvl["per_req_decode_tps"] < 0.5 * c1_decode_tps:
                    threshold_c = lvl["c"]
                    break

        out[key] = {
            "c1_decode_tps": c1_decode_tps,
            "fit": fit,                  # full fit dict (T_fixed, T_per_stream, ...)
            "aggregate_peak_c": agg_peak_c,
            "per_req_threshold_c": threshold_c,
            "levels": levels,
        }
    return out


def aggregate_by_cell(rows: list[dict]) -> dict[tuple[int, int], dict]:
    """
    Collapse JSONL rows into per-(context_size, gen_size) aggregates.

    Rates are **token-weighted**: throughput = total_tokens / total_time across
    rows in the cell. This corrects the cold-start bias that per-request means
    suffer from when some requests stop early — short-EOS rows have small
    weight in the weighted mean, so they can't drag the rate up artificially.

    Per-request decode_tps / prefill_tps stdev is kept (informational only).

    Returns: {(context, gen): {
        decode_mean, decode_std, decode_ms_per_token,
        prefill_mean, prefill_std,
        accept_mean,
        total_predicted, total_prompt,
        n  (request count)
    }}
    """
    cells: dict[tuple[int, int], dict[str, list[float]]] = {}
    for r in rows:
        key = (r.get("context_size"), r.get("gen_size"))
        if None in key:
            continue
        bucket = cells.setdefault(key, {
            "decode": [], "prefill": [], "accept": [],
            "ttft": [],
            # For token-weighted rates we need (tokens, time) pairs per request.
            "dec_tok": [], "dec_time": [],
            "pre_tok": [], "pre_time": [],
        })
        dec_tps = r.get("decode_tps")
        pre_tps = r.get("prefill_tps")
        pred_n = r.get("predicted_n") or 0
        prompt_n = r.get("prompt_n") or 0
        if dec_tps is not None:
            bucket["decode"].append(dec_tps)
            if dec_tps > 0 and pred_n > 0:
                bucket["dec_tok"].append(pred_n)
                bucket["dec_time"].append(pred_n / dec_tps)
        if pre_tps is not None:
            bucket["prefill"].append(pre_tps)
            if pre_tps > 0 and prompt_n > 0:
                bucket["pre_tok"].append(prompt_n)
                bucket["pre_time"].append(prompt_n / pre_tps)
        if r.get("ttft_ms") is not None:
            bucket["ttft"].append(r["ttft_ms"])
        if r.get("draft_n"):
            bucket["accept"].append(r["draft_accepted"] / r["draft_n"])

    out: dict[tuple[int, int], dict] = {}
    for key, bucket in cells.items():
        # Token-weighted rate = sum(tokens) / sum(time). Fall back to None when
        # we have no usable (tokens, time) data.
        def tw_rate(toks, times):
            t = sum(toks)
            s = sum(times)
            return (t / s) if (t > 0 and s > 0) else None

        decode_tw = tw_rate(bucket["dec_tok"], bucket["dec_time"])
        prefill_tw = tw_rate(bucket["pre_tok"], bucket["pre_time"])

        ttft_med = None
        if bucket["ttft"]:
            sorted_ttft = sorted(bucket["ttft"])
            ttft_med = sorted_ttft[len(sorted_ttft) // 2]

        out[key] = {
            # decode_mean / prefill_mean are now token-weighted (the user-facing
            # number). decode_arith / prefill_arith preserved for diagnostics.
            "decode_mean": decode_tw,
            "decode_arith": mean(bucket["decode"]) if bucket["decode"] else None,
            "decode_std": stdev(bucket["decode"]) if len(bucket["decode"]) > 1 else 0.0,
            "decode_ms_per_token": (1000.0 / decode_tw) if decode_tw else None,
            "prefill_mean": prefill_tw,
            "prefill_arith": mean(bucket["prefill"]) if bucket["prefill"] else None,
            "prefill_std": stdev(bucket["prefill"]) if len(bucket["prefill"]) > 1 else 0.0,
            "ttft_ms_mean": mean(bucket["ttft"]) if bucket["ttft"] else None,
            "ttft_ms_median": ttft_med,
            "ttft_ms_std": stdev(bucket["ttft"]) if len(bucket["ttft"]) > 1 else 0.0,
            "accept_mean": mean(bucket["accept"]) if bucket["accept"] else None,
            "total_predicted": sum(bucket["dec_tok"]),
            "total_prompt": sum(bucket["pre_tok"]),
            "n": max(len(bucket["decode"]), len(bucket["prefill"])),
        }
    return out


def system_footer(system: dict) -> str:
    """One-line summary for plot footers."""
    rig = system.get("rig_label", "?")
    gpus = system.get("gpus", []) or []
    sel = (system.get("selection") or {}).get("cuda_visible_devices")
    if sel:
        picks = [s.strip() for s in str(sel).split(",") if s.strip() != ""]
        selected = [g for g in gpus if str(g.get("index")) in picks]
    else:
        selected = gpus
    gpu_str = ", ".join(g.get("name", "?") for g in selected) if selected else "cpu/unknown"
    llc = system.get("llama_cpp") or {}
    commit = (llc.get("git_commit") or "?")[:8]
    flags = llc.get("build_flags") or ""
    return f"{rig} • {gpu_str} • llama.cpp {commit} {flags}".strip()


def system_diff(a: dict, b: dict) -> list[tuple[str, str, str]]:
    """Return (field, value_a, value_b) for fields that differ."""
    def gpu_label(sys_: dict) -> str:
        return ", ".join(g.get("name", "?") for g in sys_.get("gpus", []) or [])

    fields = [
        ("rig_label", a.get("rig_label"), b.get("rig_label")),
        ("gpus", gpu_label(a), gpu_label(b)),
        ("driver", _first_gpu_field(a, "driver"), _first_gpu_field(b, "driver")),
        ("cuda", _first_gpu_field(a, "cuda"), _first_gpu_field(b, "cuda")),
        ("commit", (a.get("llama_cpp") or {}).get("git_commit"),
                   (b.get("llama_cpp") or {}).get("git_commit")),
        ("build_flags", (a.get("llama_cpp") or {}).get("build_flags"),
                        (b.get("llama_cpp") or {}).get("build_flags")),
        ("cpu", (a.get("cpu") or {}).get("model"), (b.get("cpu") or {}).get("model")),
    ]
    return [(f, str(va), str(vb)) for (f, va, vb) in fields if va != vb]


def _first_gpu_field(system: dict, key: str) -> str | None:
    gpus = system.get("gpus") or []
    return gpus[0].get(key) if gpus else None
