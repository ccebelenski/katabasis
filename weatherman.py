#!/usr/bin/env python3
"""weatherman — TUI for reviewing katabasis runs. Geared for on-camera use.

Usage:
    python weatherman.py [runs_dir]

Keybindings (shown in the footer at all times):
    ↑/↓        navigate tree
    enter      open selected run
    space      pin (for A/B compare)
    1/2/3/4    switch tabs (summary / charts / raw / A/B)
    c          jump to A/B tab with pinned runs
    p          open the PNG version of the current chart (xdg-open)
    /          focus filter
    F1         toggle presenter mode
    r          refresh (also auto-refreshes on mtime change)
    q          quit
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    DataTable, Footer, Header, Input, Static, TabbedContent, TabPane, Tree,
)

import bench_data
import compare as cmp_mod

# ---------- helpers --------------------------------------------------------


def fmt_num(v, suffix: str = "", width: int = 7) -> str:
    if v is None:
        return f"{'--':>{width}}{suffix}"
    if isinstance(v, float):
        return f"{v:>{width}.1f}{suffix}"
    return f"{str(v):>{width}}{suffix}"


def trace_values(idx: bench_data.RunIndex, key: str) -> list[float]:
    """Pull a per-request trace from the run's raw.jsonl (in request order)."""
    rows = bench_data.load_jsonl(idx.path / "raw.jsonl")
    out = []
    for r in rows:
        v = r.get(key)
        if v is not None:
            out.append(float(v))
    return out


def accept_values(idx: bench_data.RunIndex) -> list[float]:
    """Per-request draft acceptance rate (0..1)."""
    rows = bench_data.load_jsonl(idx.path / "raw.jsonl")
    out = []
    for r in rows:
        n = r.get("draft_n")
        a = r.get("draft_accepted")
        if n and a is not None and n > 0:
            out.append(a / n)
    return out


# ---------- detail panels --------------------------------------------------


class SummaryPanel(Static):
    """Card view of a single run — compact rig/model strip + braille traces."""

    TRACE_W = 50
    TRACE_H = 3

    def update_run(self, idx: bench_data.RunIndex | None, presenter: bool) -> None:
        if idx is None:
            self.update("(select a run)")
            return
        if presenter:
            self.update(self._presenter_card(idx))
            return
        self.update(self._full_card(idx))

    # ---- presenter mode: hero card --------------------------------------

    def _presenter_card(self, idx: bench_data.RunIndex) -> str:
        decode_t = trace_values(idx, "decode_tps")
        prefill_t = trace_values(idx, "prefill_tps")
        dec_spark = bench_data.spark(decode_t, self.TRACE_W) if decode_t else ""
        pre_spark = bench_data.spark(prefill_t, self.TRACE_W) if prefill_t else ""
        lines = [
            "",
            "",
            f"  [b bright_white]{idx.model}[/]   [b bright_yellow]{idx.quant}[/]",
            f"  [dim]{idx.rig_label}  •  {idx.gpus_selected}[/]",
            "",
            f"  [b green]prefill[/]  {fmt_num(idx.prefill_tps_mean, ' t/s')}   [green]{pre_spark}[/]",
            f"  [b yellow]decode[/]   {fmt_num(idx.decode_tps_mean, ' t/s')}   [yellow]{dec_spark}[/]",
        ]
        if idx.accept_rate_mean is not None:
            acc = accept_values(idx)
            acc_spark = bench_data.spark([v * 100 for v in acc], self.TRACE_W, 0, 100)
            lines.append(
                f"  [b magenta]accept[/]   {fmt_num(idx.accept_rate_mean*100, '%')}   "
                f"[magenta]{acc_spark}[/]"
            )
        return "\n".join(lines)

    # ---- full mode: details + braille traces ----------------------------

    def _full_card(self, idx: bench_data.RunIndex) -> str:
        parts: list[str] = []
        parts.append(self._meta_block(idx))
        parts.append("")
        parts.append(self._headline_numbers(idx))
        rows = bench_data.load_jsonl(idx.path / "raw.jsonl")
        concurrencies = sorted({r.get("concurrency", 1) for r in rows})
        if len(concurrencies) > 1:
            parts.append("")
            parts.append(self._per_concurrency_block(rows, concurrencies))
        # Per-cell operator recommendation (#27): for each cell, does
        # concurrency actually win vs sequential? Classify by ceiling/c1.
        recommendation = self._operator_recommendation_block(rows)
        if recommendation:
            parts.append("")
            parts.append(recommendation)
        if len(idx.presets) > 1:
            parts.append("")
            parts.append(self._per_preset_block(idx))
        parts.append("")
        parts.append(self._traces_block(idx))
        return "\n".join(parts)

    def _operator_recommendation_block(self, rows: list[dict]) -> str:
        """Per-cell verdict on whether concurrency is worth it for this
        workload. Operator-actionable: cells classified as
        "sequential_wins" mean the operator is better off NOT serving
        N concurrent requests at this ctx/preset — single-stream
        delivers MORE total throughput than concurrency on this rig.
        Critical insight for production capacity-planning."""
        if not rows:
            return ""
        cells = bench_data.aggregate_ramp_by_cell(rows)
        if not cells:
            return ""
        wins, marg, loses = [], [], []
        for key in sorted(cells.keys()):
            preset, cs, gs = key
            cell = cells[key]
            regime = bench_data.classify_cell_regime(cell)
            if regime is None:
                continue
            label = f"{preset} ctx={cs} g={gs}"
            entry = (label, regime)
            if regime["regime"] == "concurrency_wins":
                wins.append(entry)
            elif regime["regime"] == "sequential_wins":
                loses.append(entry)
            else:
                marg.append(entry)
        lines = ["[b]operator recommendation (concurrency vs sequential)[/b]"]
        if loses:
            lines.append("[b red]sequential wins[/b red] — concurrency HURTS aggregate; serve one at a time:")
            for label, r in loses:
                c1 = r["ratio"]  # actually a ratio, but used for label
                lines.append(f"  [red]• {label:<32}  ceiling = {r['ratio']*100:.0f}% of c=1[/]")
        if wins:
            lines.append("[b green]concurrency wins[/b green] — serve N concurrent users for more total t/s:")
            for label, r in wins:
                lines.append(f"  [green]• {label:<32}  ceiling = {r['ratio']*100:.0f}% of c=1 ({r['label']})[/]")
        if marg:
            lines.append("[b yellow]marginal[/b yellow] — concurrency neither helps nor hurts much:")
            for label, r in marg:
                lines.append(f"  [yellow]• {label:<32}  ceiling = {r['ratio']*100:.0f}% of c=1[/]")
        if not (loses or wins or marg):
            return ""
        return "\n".join(lines)

    def _per_concurrency_block(self, rows: list[dict],
                                concurrencies: list[int]) -> str:
        """Rolling-ramp view: per (preset, ctx, gen) cell, show the c=1
        baseline, aggregate peak, per-req threshold, and fit deviation
        summary. Replaces the closed-loop per-N workgroup table."""
        cells = bench_data.aggregate_ramp_by_cell(rows)
        if not cells:
            return ""
        lines = ["[b]per-cell ramp[/]  "
                 "[dim](c=1 baseline · aggregate peak · per-req <50% c=1 threshold · "
                 "max |Δ| from c1/c fit)[/]"]
        for key in sorted(cells.keys()):
            preset, cs, gs = key
            cell = cells[key]
            c1 = cell.get("c1_decode_tps")
            if c1 is None:
                lines.append(f"  [b magenta]{preset}[/] ctx={cs} gen={gs}  "
                             f"[dim](no baseline)[/]")
                continue
            peak_c = cell.get("aggregate_peak_c")
            thresh_c = cell.get("per_req_threshold_c")
            agg_at_peak = next((lvl["aggregate_decode_tps"] for lvl in cell["levels"]
                                if lvl["c"] == peak_c), None)
            devs = [lvl["deviation"] for lvl in cell["levels"]
                    if lvl.get("deviation") is not None and lvl["c"] > 1]
            max_dev_pct = (max(abs(d) for d in devs) * 100) if devs else None
            bits = [
                f"  [b magenta]{preset}[/] ctx={cs} g={gs}",
                f"c=1 [yellow]{c1:6.1f}[/] t/s",
                (f"agg peak [b bright_cyan]{agg_at_peak:6.1f}[/] @c={peak_c}"
                 if agg_at_peak is not None else "agg peak --"),
                (f"<50% @c={thresh_c}" if thresh_c is not None else "<50% n/a"),
                (f"max |Δ| [blue]{max_dev_pct:4.1f}[/]%"
                 if max_dev_pct is not None else "Δ n/a"),
                (f"[dim](ceil≈{cell['fit']['aggregate_ceiling']:.0f} t/s)[/]"
                 if cell.get("fit") else "[dim](no fit)[/]"),
            ]
            lines.append("   ".join(bits))
        return "\n".join(lines)

    def _per_preset_block(self, idx: bench_data.RunIndex) -> str:
        """When multiple presets ran, show a per-preset stats row each."""
        from statistics import mean as _mean
        rows = bench_data.load_jsonl(idx.path / "raw.jsonl")
        lines = ["[b]per-preset[/]"]
        for preset in idx.presets:
            sub = [r for r in rows if r.get("preset") == preset]
            if not sub:
                continue
            dec = [r["decode_tps"] for r in sub if r.get("decode_tps")]
            pre = [r["prefill_tps"] for r in sub if r.get("prefill_tps")]
            ttft = [r["ttft_ms"] for r in sub if r.get("ttft_ms") is not None]
            acc = [r["draft_accepted"] / r["draft_n"] for r in sub
                   if r.get("draft_n") and r.get("draft_accepted") is not None]
            bits = [
                f"  [b magenta]{preset:>8}[/]",
                f"prefill [green]{_mean(pre):7.1f}[/] t/s" if pre else "prefill   -- t/s",
                f"decode [yellow]{_mean(dec):6.1f}[/] t/s" if dec else "decode   -- t/s",
            ]
            if ttft:
                bits.append(f"TTFT [blue]{_mean(ttft):6.0f}[/] ms")
            if acc:
                bits.append(f"accept [magenta]{_mean(acc)*100:5.1f}[/]%")
            bits.append(f"[dim]({len(sub)} rows)[/]")
            lines.append("   ".join(bits))
        return "\n".join(lines)

    def _meta_block(self, idx: bench_data.RunIndex) -> str:
        # Two-column compact layout — label width fixed for vertical alignment.
        lab = lambda s: f"[b magenta]{s:>9}[/]"
        system = bench_data.load_system(idx.path)
        llc = system.get("llama_cpp") or {}
        sel = system.get("selection") or {}
        cfg = system.get("config") or {}
        sweep = cfg.get("sweep") or {}
        server_args = cfg.get("server_args") or {}
        grouped, ungrouped = bench_data.categorize_server_args(server_args)

        def row(label: str, value: str) -> str:
            return f"{lab(label)}  {value}"

        lines: list[str] = [
            row("model", f"[bright_white]{idx.model}[/]  [b yellow]{idx.quant}[/]"),
            row("draft", idx.draft_model or "[dim](none)[/]"),
            row("spec", idx.spec_type or "[dim](none)[/]"),
            row("rig", f"{idx.rig_label}  [dim]•[/]  {idx.gpus_selected}"),
            row("commit", f"[dim]{idx.commit or '?'}[/]"
                + (f"  [dim]build:{llc.get('build_flags')}[/]" if llc.get("build_flags") else "")
                + f"  [dim]{bench_data.format_timestamp_local(idx.timestamp)}[/]"),
            row("path", f"[dim]{idx.path}[/]"),
        ]

        # Selection (CUDA / tensor_split / main_gpu) — only if a value is set.
        sel_bits = []
        if sel.get("cuda_visible_devices") is not None:
            sel_bits.append(f"cvd={sel['cuda_visible_devices']}")
        if sel.get("tensor_split"):
            sel_bits.append(f"split={sel['tensor_split']}")
        if sel.get("main_gpu") is not None:
            sel_bits.append(f"main={sel['main_gpu']}")
        if sel_bits:
            lines.append(row("select", f"[dim]{'  '.join(sel_bits)}[/]"))

        # Sweep grid — what was actually run.
        if sweep:
            lines.append(row("sweep",
                f"ctx={bench_data.get_context_sizes(sweep)}  "
                f"gen={sweep.get('gen_sizes')}  "
                f"rounds={sweep.get('rounds')}  "
                f"warmup={sweep.get('warmup_rounds')}  "
                f"presets={bench_data.get_prompt_presets(sweep)}"
            ))

        # Categorized server args.
        def fmt_group(entries: list[tuple[str, str]]) -> str:
            return "  ".join(
                f"[bright_white]{k}[/]={v}" if v else f"[bright_white]{k}[/]"
                for k, v in entries
            )

        for name, entries in grouped.items():
            if entries:
                lines.append(row(name, fmt_group(entries)))
        if ungrouped:
            lines.append(row("other", fmt_group(ungrouped)))

        return "\n".join(lines)

    def _headline_numbers(self, idx: bench_data.RunIndex) -> str:
        bits = [
            f"[b green]prefill[/]  {fmt_num(idx.prefill_tps_mean, ' t/s')}",
            f"[b yellow]decode[/]   {fmt_num(idx.decode_tps_mean, ' t/s')}",
        ]
        if idx.ttft_ms_mean is not None:
            bits.append(f"[b blue]TTFT[/]   {fmt_num(idx.ttft_ms_mean, ' ms', 6)}")
        if idx.accept_rate_mean is not None:
            bits.append(f"[b magenta]accept[/]   {fmt_num(idx.accept_rate_mean*100, '%')}")
        bits.append(f"[dim]{idx.rows} rows[/]")
        return "    ".join(bits)

    def _traces_block(self, idx: bench_data.RunIndex) -> str:
        """Per-request braille traces — same visual language as katabasis."""
        decode_t = trace_values(idx, "decode_tps")
        prefill_t = trace_values(idx, "prefill_tps")
        ttft_t = trace_values(idx, "ttft_ms")
        out: list[str] = []
        # Auto-scale (no lo=0) so small round-to-round variance is visible —
        # the cur/mean labels already give absolute scale.
        if prefill_t:
            out.append(self._trace_chart("prefill", prefill_t, "t/s",
                                         "bright_green"))
            out.append("")
        if decode_t:
            out.append(self._trace_chart("decode", decode_t, "t/s",
                                         "bright_yellow"))
        if ttft_t:
            out.append("")
            out.append(self._trace_chart("TTFT", ttft_t, "ms",
                                         "bright_blue"))
        if idx.accept_rate_mean is not None:
            acc = [v * 100 for v in accept_values(idx)]
            if acc:
                out.append("")
                out.append(self._trace_chart("accept", acc, "%",
                                             "bright_magenta", lo=0, hi=100))
        if not out:
            out.append("[dim](no raw.jsonl rows to plot)[/]")
        return "\n".join(out)

    def _trace_chart(self, label: str, vals: list[float], unit: str,
                     style: str, lo: float | None = None,
                     hi: float | None = None) -> str:
        from statistics import mean as _mean
        bars = bench_data.braille_bars(vals, self.TRACE_W, self.TRACE_H, lo=lo, hi=hi)
        cur = vals[-1]
        mn = _mean(vals)
        head = (f"[dim]{label:>7}[/] [{style}]{bars[0]}[/]   "
                f"cur {cur:7.1f} {unit}   [dim]mean {mn:7.1f} {unit}[/]")
        rest = [f"[dim]{'':>7}[/] [{style}]{b}[/]" for b in bars[1:]]
        return "\n".join([head] + rest)


class ChartsPanel(Static):
    """Per-run charts. When the run has multiple presets, render one set of
    facet charts per preset PLUS an overlay chart with one line per preset."""

    def update_run(self, idx: bench_data.RunIndex | None, width: int, height: int) -> None:
        if idx is None:
            self.update("(select a run)")
            return
        rows = bench_data.load_jsonl(idx.path / "raw.jsonl")
        if not rows:
            self.update("(no raw.jsonl)")
            return
        presets = sorted({r["preset"] for r in rows if r.get("preset")})
        concurrencies = sorted({r.get("concurrency", 1) for r in rows})

        w = max(50, width - 6)
        h = max(10, (height - 6) // 2)

        parts: list[str] = []

        # ---- ramp curves: measured + c1/c fit overlay --------------------
        cells = bench_data.aggregate_ramp_by_cell(rows)
        if cells and len(concurrencies) > 1:
            parts.append("[b cyan]── ramp curves: measured vs c1/c fit ──[/]")
            # One overlay chart for per-req, one for aggregate. Each cell
            # contributes two series (measured + predicted), keyed by the
            # cell label.
            per_req_series = []
            agg_series = []
            for key in sorted(cells.keys()):
                preset, cs, gs = key
                cell = cells[key]
                c1 = cell.get("c1_decode_tps")
                fit = cell.get("fit")
                if c1 is None:
                    continue
                cs_x = [lvl["c"] for lvl in cell["levels"]
                        if lvl["per_req_decode_tps"] is not None]
                if not cs_x:
                    continue
                per_req = [lvl["per_req_decode_tps"] for lvl in cell["levels"]
                           if lvl["per_req_decode_tps"] is not None]
                agg = [lvl["aggregate_decode_tps"] for lvl in cell["levels"]
                       if lvl["per_req_decode_tps"] is not None]
                lbl = f"{preset} ctx={cs} g={gs}"
                per_req_series.append({"label": f"{lbl} measured", "x": cs_x, "y": per_req})
                agg_series.append({"label": f"{lbl} measured", "x": cs_x, "y": agg})
                if fit is not None:
                    pred_x = list(range(1, max(cs_x) + 1))
                    pred_per = [bench_data.predict_ramp_per_req(c, fit)
                                for c in pred_x]
                    pred_agg = [c * p if p is not None else None
                                for c, p in zip(pred_x, pred_per)]
                    per_req_series.append({"label": f"{lbl} predicted",
                                           "x": pred_x, "y": pred_per})
                    agg_series.append({"label": f"{lbl} predicted",
                                       "x": pred_x, "y": pred_agg})
            if per_req_series:
                parts.append(bench_data.line_chart(
                    per_req_series, width=w, height=h,
                    x_label="concurrency", y_label="per-req t/s",
                    title="per-request throughput vs c"))
            if agg_series:
                parts.append(bench_data.line_chart(
                    agg_series, width=w, height=h,
                    x_label="concurrency", y_label="aggregate t/s",
                    title="aggregate throughput vs c"))

            # TTFT growth across c — uses in_steady_state rows only.
            ttft_series = []
            from statistics import mean as _mean
            for key in sorted(cells.keys()):
                preset, cs, gs = key
                cell = cells[key]
                cs_x = [lvl["c"] for lvl in cell["levels"]
                        if lvl.get("ttft_ms_mean") is not None]
                tmean = [lvl["ttft_ms_mean"] for lvl in cell["levels"]
                         if lvl.get("ttft_ms_mean") is not None]
                tp95 = [lvl["ttft_ms_p95"] for lvl in cell["levels"]
                        if lvl.get("ttft_ms_p95") is not None]
                lbl = f"{preset} ctx={cs} g={gs}"
                if cs_x:
                    ttft_series.append({"label": f"{lbl} mean", "x": cs_x, "y": tmean})
                if tp95:
                    ttft_series.append({"label": f"{lbl} p95", "x": cs_x, "y": tp95})
            if ttft_series:
                parts.append(bench_data.line_chart(
                    ttft_series, width=w, height=h,
                    x_label="concurrency", y_label="TTFT ms",
                    title="TTFT vs concurrency"))

        # The per-preset facets below show the c=1 single-user baseline
        # only — multi-c story is in the ramp curves above.
        if len(concurrencies) > 1:
            preset_rows = [r for r in rows
                           if r.get("concurrency") == 1 and r.get("in_steady_state")]
            parts.append("[dim](per-preset facets below: c=1 steady-state baseline "
                         "— multi-c story is in the ramp curves above)[/]")
        else:
            preset_rows = rows

        # ---- combined overlay (one line per preset) — only when >1 preset ----
        if len(presets) > 1:
            parts.append(f"[b cyan]── combined (overlay across presets) ──[/]")
            from statistics import mean as _mean
            # decode_vs_gen_by_preset: x = gen, y = mean decode across ctx
            ds = []
            for preset in presets:
                sub = [r for r in preset_rows if r["preset"] == preset]
                sub_cells = bench_data.aggregate_by_cell(sub)
                xs, ys = [], []
                for gs in sorted({g for (_, g) in sub_cells}):
                    vals = [c["decode_mean"] for k, c in sub_cells.items()
                            if k[1] == gs and c["decode_mean"] is not None]
                    if vals:
                        xs.append(gs); ys.append(_mean(vals))
                if xs:
                    ds.append({"label": preset, "x": xs, "y": ys})
            if ds:
                parts.append(bench_data.line_chart(
                    ds, width=w, height=h,
                    x_label="gen tokens", y_label="decode t/s",
                    title="decode vs gen — one line per preset"))
            # prefill_vs_context_by_preset
            ps = []
            for preset in presets:
                sub = [r for r in preset_rows if r["preset"] == preset]
                sub_cells = bench_data.aggregate_by_cell(sub)
                xs, ys = [], []
                for cs in sorted({c for (c, _) in sub_cells}):
                    vals = [c2["prefill_mean"] for k, c2 in sub_cells.items()
                            if k[0] == cs and c2["prefill_mean"] is not None]
                    if vals:
                        xs.append(cs); ys.append(_mean(vals))
                if xs:
                    ps.append({"label": preset, "x": xs, "y": ys})
            if ps:
                parts.append(bench_data.line_chart(
                    ps, width=w, height=h,
                    x_label="context tokens", y_label="prefill t/s",
                    title="prefill vs ctx — one line per preset", x_log=True))

            # ttft_vs_context_by_preset
            ts = []
            for preset in presets:
                sub = [r for r in preset_rows if r["preset"] == preset]
                sub_cells = bench_data.aggregate_by_cell(sub)
                xs, ys = [], []
                for cs in sorted({c for (c, _) in sub_cells}):
                    vals = [c2.get("ttft_ms_mean") for k, c2 in sub_cells.items()
                            if k[0] == cs and c2.get("ttft_ms_mean") is not None]
                    if vals:
                        xs.append(cs); ys.append(_mean(vals))
                if xs:
                    ts.append({"label": preset, "x": xs, "y": ys})
            if ts:
                parts.append(bench_data.line_chart(
                    ts, width=w, height=h,
                    x_label="context tokens", y_label="TTFT ms",
                    title="TTFT vs ctx — one line per preset", x_log=True))

        # ---- per-preset facets ---------------------------------------------
        for preset in presets:
            sub = [r for r in preset_rows if r["preset"] == preset]
            sub_cells = bench_data.aggregate_by_cell(sub)
            if not sub_cells:
                continue

            # Skip the section header when there's only one preset — no need
            # to disambiguate.
            if len(presets) > 1:
                parts.append(f"[b magenta]── preset: {preset}  ({len(sub)} rows) ──[/]")

            # decode vs gen — one series per context size
            decode_series = []
            for cs in sorted({c for (c, _) in sub_cells}):
                xs, ys = [], []
                for (c, gs) in sorted(sub_cells):
                    if c != cs:
                        continue
                    v = sub_cells[(c, gs)]["decode_mean"]
                    if v is not None:
                        xs.append(gs); ys.append(v)
                if xs:
                    decode_series.append({"label": f"ctx={cs}", "x": xs, "y": ys})

            # prefill vs context — one series per gen size, log x
            prefill_series = []
            for gs in sorted({g for (_, g) in sub_cells}):
                xs, ys = [], []
                for (cs, g) in sorted(sub_cells):
                    if g != gs:
                        continue
                    v = sub_cells[(cs, g)]["prefill_mean"]
                    if v is not None:
                        xs.append(cs); ys.append(v)
                if xs:
                    prefill_series.append({"label": f"g={gs}", "x": xs, "y": ys})

            # TTFT vs context — one series per gen size, log x (and log y is
            # natural since TTFT scales linearly with ctx over orders of magnitude)
            ttft_series = []
            for gs in sorted({g for (_, g) in sub_cells}):
                xs, ys = [], []
                for (cs, g) in sorted(sub_cells):
                    if g != gs:
                        continue
                    v = sub_cells[(cs, g)].get("ttft_ms_mean")
                    if v is not None:
                        xs.append(cs); ys.append(v)
                if xs:
                    ttft_series.append({"label": f"g={gs}", "x": xs, "y": ys})

            if prefill_series:
                parts.append(bench_data.line_chart(
                    prefill_series, width=w, height=h,
                    x_label="context tokens", y_label="prefill t/s",
                    title=f"prefill vs context  ({preset})", x_log=True))
            if decode_series:
                parts.append(bench_data.line_chart(
                    decode_series, width=w, height=h,
                    x_label="gen tokens", y_label="decode t/s",
                    title=f"decode vs generation length  ({preset})"))
            if ttft_series:
                parts.append(bench_data.line_chart(
                    ttft_series, width=w, height=h,
                    x_label="context tokens", y_label="TTFT ms",
                    title=f"TTFT vs context  ({preset})", x_log=True))

        # ---- draft acceptance — single series, time-ordered ----------------
        accept_rows = [r for r in rows if r.get("draft_n")]
        if accept_rows:
            rates = [(r["draft_accepted"] / r["draft_n"]) * 100 for r in accept_rows
                     if r.get("draft_accepted") is not None and r["draft_n"]]
            if rates:
                from statistics import mean as _mean
                bar_w = max(30, min(w // 2, len(rates)))
                bars = bench_data.braille_bars(rates, bar_w, 4, lo=0, hi=100)
                hdr = (f"[b]draft acceptance per request[/b]   "
                       f"[dim]mean {_mean(rates):.1f}%  •  "
                       f"min {min(rates):.1f}%  •  max {max(rates):.1f}%[/]")
                parts.append(hdr + "\n" +
                             "\n".join(f"  [magenta]{row}[/]" for row in bars))
        self.update("\n\n".join(parts) if parts else "(no chartable data)")


class RawPanel(Vertical):
    """Scrollable DataTable of the JSONL rows."""

    def compose(self) -> ComposeResult:
        self.table = DataTable(zebra_stripes=True, cursor_type="row")
        self.table.add_columns(
            "round", "preset", "ctx", "gen", "c",
            "prefill t/s", "decode t/s", "TTFT ms",
            "draft_n", "accepted", "wall s",
        )
        yield self.table

    def update_run(self, idx: bench_data.RunIndex | None) -> None:
        self.table.clear()
        if idx is None:
            return
        rows = bench_data.load_jsonl(idx.path / "raw.jsonl")
        for r in rows:
            ttft = r.get("ttft_ms")
            self.table.add_row(
                str(r.get("round", "")),
                str(r.get("preset", "")),
                str(r.get("context_size", "")),
                str(r.get("gen_size", "")),
                str(r.get("concurrency", 1)),
                f"{r.get('prefill_tps') or 0:.1f}",
                f"{r.get('decode_tps') or 0:.1f}",
                f"{ttft:.0f}" if ttft is not None else "-",
                str(r.get("draft_n") or "-"),
                str(r.get("draft_accepted") or "-"),
                f"{r.get('wall_s') or 0:.2f}",
            )


class SearchPanel(Vertical):
    """Cross-run event search (#19). Scans all events.log files under
    the runs directory for the query string; matches against message
    text and structured field values. Result row: run + time + type +
    msg. Lets the operator find patterns across the tree without
    drilling into each run's Events tab — e.g. "find all runs that
    mentioned 'Host memory'" surfaces every CPU-spillover incident.

    Designed for triage and forensics, not for browsing. Limited to
    200 results to keep the UI responsive.
    """

    def compose(self) -> ComposeResult:
        yield Input(placeholder="search events across all runs (msg + fields)…",
                    id="search_input")
        self.table = DataTable(zebra_stripes=True, cursor_type="row")
        self.table.add_columns("run", "time", "level", "type", "message")
        yield self.table

    def update_for_query(self, runs_dir: Path, query: str) -> None:
        self.table.clear()
        if not query or not query.strip():
            return
        results = bench_data.search_events_across_runs(
            runs_dir, query.strip(), limit=200,
        )
        for r in results:
            level = r.get("level", "info")
            color = EventsPanel.LEVEL_STYLES.get(level, "cyan")
            self.table.add_row(
                Text(r.get("run_name", "")[:30], style="dim"),
                r.get("ts_local", ""),
                Text(level, style=color),
                r.get("type", ""),
                Text(r.get("msg", "")[:120], style=color),
            )


class EventsPanel(Vertical):
    """Scrollable per-run timeline from events.log (#17 + #18 Tier 1).

    Shows decision-level events kata recorded during the run in
    chronological order: cell starts, level completions, knee labels,
    termination reasons, fit results, VRAM warnings. Color-coded by
    level (red=warn/error, green=peak/success, yellow=transition,
    cyan=info)."""

    LEVEL_STYLES = {
        "info": "cyan",
        "transition": "yellow",
        "peak": "bright_green",
        "success": "green",
        "warn": "red",
        "error": "bold red",
    }

    def compose(self) -> ComposeResult:
        self.table = DataTable(zebra_stripes=True, cursor_type="row")
        self.table.add_columns("time", "level", "type", "message")
        yield self.table

    def update_run(self, idx: bench_data.RunIndex | None) -> None:
        self.table.clear()
        if idx is None:
            return
        events = bench_data.load_events(idx.path)
        for ev in events:
            ts = ev.get("ts_local", "")
            level = ev.get("level", "info")
            ev_type = ev.get("type", "")
            msg = ev.get("msg", "")
            style = self.LEVEL_STYLES.get(level, "cyan")
            # Style applied to the message column (most-visible cue) plus
            # the level column for quick scanning.
            self.table.add_row(
                ts,
                Text(level, style=style),
                ev_type,
                Text(msg, style=style),
            )


class ABPanel(Static):
    """Overlay charts + system-diff table across pinned runs."""

    @staticmethod
    def _termination_per_run(pinned: list[bench_data.RunIndex]) -> list[dict]:
        """Pull termination events from each run's events.log, keyed by
        (preset, ctx, gen). Used by both the decision-diff block and
        the coverage banner."""
        out = []
        for r in pinned:
            terms: dict = {}
            current_cell = None
            for ev in bench_data.load_events(r.path):
                if ev.get("type") == "cell_start":
                    current_cell = (ev.get("preset"), ev.get("context_size"),
                                    ev.get("gen_size"))
                elif ev.get("type") == "termination" and current_cell is not None:
                    terms[current_cell] = ev.get("stop_reason", "?")
            out.append(terms)
        return out

    @staticmethod
    def _coverage_banner(pinned: list[bench_data.RunIndex],
                         cells_per_run: list[dict]) -> str:
        """Top-of-A/B banner showing the per-run cell counts + how many
        are in common. Sets the operator's expectation honestly before
        they read any numbers."""
        common, uniques = bench_data.intersect_run_cells(cells_per_run)
        parts = ["[b]coverage[/b]"]
        for p, cells, uniq in zip(pinned, cells_per_run, uniques):
            parts.append(
                f"  {p.path.name[:30]:<32}  "
                f"{len(cells)} cells  ([yellow]{len(uniq)} unique to this run[/])"
            )
        parts.append(f"  [bright_cyan]{len(common)} cells in common — "
                     f"directly comparable[/]")
        return "\n".join(parts)

    @staticmethod
    def _decision_diff_block(pinned: list[bench_data.RunIndex],
                              cells_per_run: list[dict]) -> str:
        """Termination-reason matrix over INTERSECTING cells only, with
        a uniqueness footer naming what each run measured that the other(s)
        didn't. Replaces the old all-cells-with-blanks version — that one
        conflated "not measured" with "ran to max_c", per audit #10 + the
        2026-06-06 ctx-sweep change.
        """
        common, uniques = bench_data.intersect_run_cells(cells_per_run)
        per_run_terminations = ABPanel._termination_per_run(pinned)
        if not any(per_run_terminations):
            return ""
        lines = ["[b]decision diff — per-cell termination (common cells only)[/b]"]
        labels = [p.path.name[:30] for p in pinned]
        header = f"  {'cell':<32}  " + "  ".join(f"{lbl:<30}" for lbl in labels)
        lines.append(header)
        if not common:
            lines.append("  [dim](no cells in common between pinned runs)[/dim]")
        for cell in sorted(common):
            preset, cs, gs = cell
            cell_str = f"{preset} ctx={cs} g={gs}"
            cols = []
            for terms in per_run_terminations:
                stop = terms.get(cell, "—")
                cols.append(stop)
            distinct = {c for c in cols if c != "—"}
            if len(distinct) > 1:
                row = (f"  [yellow]{cell_str:<32}[/]  "
                       + "  ".join(f"[yellow]{c:<30}[/]" for c in cols))
            else:
                row = (f"  {cell_str:<32}  "
                       + "  ".join(f"[dim]{c:<30}[/]" for c in cols))
            lines.append(row)
        # Uniqueness footer: name what each run had that the others didn't.
        for p, uniq in zip(pinned, uniques):
            if not uniq:
                continue
            cells_str = ", ".join(
                f"{preset} ctx={cs} g={gs}"
                for (preset, cs, gs) in sorted(uniq)
            )
            lines.append(f"  [dim]cells unique to {p.path.name[:30]}: "
                         f"{cells_str}[/]")
        if any(uniques):
            lines.append("")
        if len(common) > 0:
            lines.append("  [dim](yellow = runs disagree on termination → "
                         "different operating regime)[/]")
        return "\n".join(lines)

    @staticmethod
    def _interpolated_ctx_table(pinned: list[bench_data.RunIndex],
                                 cells_per_run: list[dict]) -> str:
        """Build a unified-ctx comparison table across all pinned runs.
        For each (preset, gen) combination in the union of measured
        configs, list every ctx that any run measured. For each (ctx,
        run) cell, show the c=1 baseline and fit ceiling — either as
        measured (●) or linearly interpolated (◇) from bracketing
        measured ctx values for that run. ctx values outside a run's
        measured range get '—' (no extrapolation).

        Operator value: rigorous cross-rig numerical comparison even
        when sweep matrices don't fully overlap. The ◇/● markers
        keep the operator honest about which values are direct
        measurements vs projections.
        """
        # Discover all (preset, gen) and all ctx values present anywhere.
        all_pgs: set[tuple] = set()
        all_ctx: set[int] = set()
        for cells in cells_per_run:
            for (preset, cs, gs) in cells.keys():
                all_pgs.add((preset, gs))
                all_ctx.add(cs)
        if not all_pgs or not all_ctx:
            return ""

        lines = ["[b]cross-ctx interpolated comparison — c=1 baseline · fit ceiling[/b]",
                 "  [dim]● = measured · ◇ = linearly interpolated · — = outside measured range[/]"]
        labels = [p.path.name[:24] for p in pinned]

        for preset, gs in sorted(all_pgs):
            lines.append(f"\n  [bold]{preset} gen={gs}[/bold]")
            header = (f"    {'ctx':>6}  "
                      + "  ".join(f"{lbl:<28}" for lbl in labels))
            lines.append(header)
            for ctx in sorted(all_ctx):
                cols = []
                for cells in cells_per_run:
                    key = (preset, ctx, gs)
                    if key in cells:
                        cell = cells[key]
                        c1 = cell.get("c1_decode_tps")
                        fit = cell.get("fit")
                        ceiling = fit.get("aggregate_ceiling") if fit else None
                        c1_s = f"{c1:.1f}" if c1 is not None else "?"
                        ceil_s = f"{ceiling:.1f}" if ceiling is not None else "?"
                        regime = bench_data.classify_cell_regime(cell)
                        regime_s = (f" [{regime['color']}]{regime['label']}[/]"
                                    if regime else "")
                        cols.append(f"[bright_white]●[/] c1={c1_s} ceil={ceil_s}"
                                    + regime_s)
                    else:
                        proj = bench_data.interpolate_cell_metrics_at_ctx(
                            cells, preset, ctx, gs)
                        if proj is None:
                            cols.append("[dim]—[/]".ljust(28))
                            continue
                        c1 = proj.get("c1_decode_tps")
                        fit = proj.get("fit")
                        ceiling = fit.get("aggregate_ceiling") if fit else None
                        c1_s = f"{c1:.1f}" if c1 is not None else "?"
                        ceil_s = f"{ceiling:.1f}" if ceiling is not None else "?"
                        regime = bench_data.classify_cell_regime(proj)
                        regime_s = (f" [{regime['color']}]{regime['label']}[/]"
                                    if regime else "")
                        cols.append(f"[yellow]◇[/] c1={c1_s} ceil={ceil_s}"
                                    + regime_s)
                # Compute ratio across the first two runs if both have data.
                ratio_str = ""
                if len(cells_per_run) == 2:
                    def get_c1(cells):
                        key = (preset, ctx, gs)
                        if key in cells:
                            return cells[key].get("c1_decode_tps")
                        proj = bench_data.interpolate_cell_metrics_at_ctx(
                            cells, preset, ctx, gs)
                        return proj.get("c1_decode_tps") if proj else None
                    a = get_c1(cells_per_run[0])
                    b = get_c1(cells_per_run[1])
                    if a and b and b > 0:
                        ratio_str = f"  [bright_cyan]{a/b:.2f}×[/]"
                lines.append(f"    {ctx:>6}  " + "  ".join(c.ljust(28) for c in cols)
                             + ratio_str)
        return "\n".join(lines)

    def update_pinned(self, pinned: list[bench_data.RunIndex], width: int, height: int) -> None:
        if len(pinned) < 2:
            self.update("[dim]pin two or more runs (space/b) and press [b]c[/b] to compare[/dim]")
            return

        loads = [cmp_mod.load(p.path) for p in pinned]
        # Per-run raw rows for concurrency-aware overlays.
        rows_by_run = [bench_data.load_jsonl(p.path / "raw.jsonl") for p in pinned]
        w = max(50, width - 6)
        h = max(10, (height - 18) // 2)

        from statistics import mean as _mean
        parts: list[str] = []

        # ---- Decode vs gen (one line per run, mean across ctx) -----------
        decode_series = []
        for l in loads:
            context_sizes = sorted({cs for (cs, _) in l.cells.keys()})
            gen_sizes = sorted({gs for (_, gs) in l.cells.keys()})
            xs, ys = [], []
            for gs in gen_sizes:
                vals = [l.cells[(cs, gs)]["decode_mean"] for cs in context_sizes
                        if (cs, gs) in l.cells
                        and l.cells[(cs, gs)]["decode_mean"] is not None]
                if vals:
                    xs.append(gs); ys.append(_mean(vals))
            if xs:
                decode_series.append({"label": l.label, "x": xs, "y": ys})
        if decode_series:
            parts.append(bench_data.line_chart(
                decode_series, width=w, height=h,
                x_label="gen tokens", y_label="decode t/s",
                title="decode (mean across ctx)"))

        # ---- Prefill vs context (one line per run, log x) -----------------
        prefill_series = []
        for l in loads:
            gen_sizes = sorted({gs for (_, gs) in l.cells.keys()})
            context_sizes = sorted({cs for (cs, _) in l.cells.keys()})
            xs, ys = [], []
            for cs in context_sizes:
                vals = [l.cells[(cs, gs)]["prefill_mean"] for gs in gen_sizes
                        if (cs, gs) in l.cells
                        and l.cells[(cs, gs)]["prefill_mean"] is not None]
                if vals:
                    xs.append(cs); ys.append(_mean(vals))
            if xs:
                prefill_series.append({"label": l.label, "x": xs, "y": ys})
        if prefill_series:
            parts.append(bench_data.line_chart(
                prefill_series, width=w, height=h,
                x_label="context tokens", y_label="prefill t/s",
                title="prefill vs ctx (mean across gen)", x_log=True))

        # ---- TTFT vs context (one line per run, log x; non-zero-base ok) --
        ttft_series = []
        for l in loads:
            gen_sizes = sorted({gs for (_, gs) in l.cells.keys()})
            context_sizes = sorted({cs for (cs, _) in l.cells.keys()})
            xs, ys = [], []
            for cs in context_sizes:
                vals = [l.cells[(cs, gs)].get("ttft_ms_mean") for gs in gen_sizes
                        if (cs, gs) in l.cells
                        and l.cells[(cs, gs)].get("ttft_ms_mean") is not None]
                if vals:
                    xs.append(cs); ys.append(_mean(vals))
            if xs:
                ttft_series.append({"label": l.label, "x": xs, "y": ys})
        if ttft_series:
            parts.append(bench_data.line_chart(
                ttft_series, width=w, height=h,
                x_label="context tokens", y_label="TTFT ms",
                title="TTFT vs ctx (mean across gen)", x_log=True))

        # ---- Ramp overlay across runs (per-cell measured curves) ---------
        all_cs = set()
        for run_rows in rows_by_run:
            all_cs |= {r.get("concurrency", 1) for r in run_rows}
        if len(all_cs) > 1:
            agg_series = []
            per_req_series = []
            ttftc_series = []
            for l, run_rows in zip(loads, rows_by_run):
                cells = bench_data.aggregate_ramp_by_cell(run_rows)
                # If only one cell per run, label just by run; if multiple,
                # disambiguate per-cell.
                for key in sorted(cells.keys()):
                    preset, cs, gs = key
                    cell = cells[key]
                    if cell.get("c1_decode_tps") is None:
                        continue
                    levels_with_data = [lvl for lvl in cell["levels"]
                                        if lvl["per_req_decode_tps"] is not None]
                    if not levels_with_data:
                        continue
                    cs_x = [lvl["c"] for lvl in levels_with_data]
                    lbl = (l.label if len(cells) == 1
                           else f"{l.label} {preset} ctx={cs} g={gs}")
                    per_req_series.append({"label": lbl, "x": cs_x,
                                           "y": [lvl["per_req_decode_tps"]
                                                 for lvl in levels_with_data]})
                    agg_series.append({"label": lbl, "x": cs_x,
                                       "y": [lvl["aggregate_decode_tps"]
                                             for lvl in levels_with_data]})
                    ttft_x = [lvl["c"] for lvl in cell["levels"]
                              if lvl.get("ttft_ms_mean") is not None]
                    if ttft_x:
                        ttftc_series.append({"label": lbl, "x": ttft_x,
                                             "y": [lvl["ttft_ms_mean"]
                                                   for lvl in cell["levels"]
                                                   if lvl.get("ttft_ms_mean") is not None]})
            if per_req_series:
                parts.append(bench_data.line_chart(
                    per_req_series, width=w, height=h,
                    x_label="concurrency", y_label="per-req t/s",
                    title="per-request throughput vs c"))
            if agg_series:
                parts.append(bench_data.line_chart(
                    agg_series, width=w, height=h,
                    x_label="concurrency", y_label="aggregate t/s",
                    title="aggregate throughput vs c"))
            if ttftc_series:
                parts.append(bench_data.line_chart(
                    ttftc_series, width=w, height=h,
                    x_label="concurrency", y_label="TTFT ms",
                    title="TTFT vs concurrency"))

        # ---- Cross-run coverage + decision diff + interpolated table ------
        # (#26: intersect + interpolation for matrix-mismatch honesty)
        # Build the cells dicts once (used by all three sections).
        cells_per_run = [
            bench_data.aggregate_ramp_by_cell(rows) for rows in rows_by_run
        ]
        parts.insert(0, self._coverage_banner(pinned, cells_per_run))
        # Decision diff over intersecting cells (with uniqueness footer).
        decision_block = self._decision_diff_block(pinned, cells_per_run)
        if decision_block:
            parts.append(decision_block)
        # Interpolated cross-ctx table — numerical comparison even when
        # ctx sweeps don't fully overlap. ● = measured, ◇ = interpolated.
        interp_block = self._interpolated_ctx_table(pinned, cells_per_run)
        if interp_block:
            parts.append(interp_block)

        # ---- System diff table + speedup ----------------------------------
        if len(loads) == 2:
            diff = bench_data.system_diff(loads[0].system, loads[1].system)
            sys_block = ["[b]system diff[/b]"]
            if not diff:
                sys_block.append("  (none captured)")
            else:
                sys_block.append(f"  {'field':<14}  {'A':<30}  B")
                for f, va, vb in diff:
                    va_s = va if len(va) < 28 else va[:25] + "..."
                    vb_s = vb if len(vb) < 28 else vb[:25] + "..."
                    sys_block.append(f"  {f:<14}  {va_s:<30}  {vb_s}")

            speedup = cmp_mod.compute_speedup(loads)
            safe = cmp_mod.configs_equivalent(loads)
            if speedup is not None:
                tag = ("[b yellow]speedup[/b yellow]" if safe
                       else "[b red]ratio (apples-to-oranges)[/b red]")
                sys_block.append(f"\n  {tag}: [b]{speedup:.2f}×[/b]")
            parts.append("\n".join(sys_block))

        self.update("\n\n".join(parts) if parts else "(no chartable data)")


# ---------- main app -------------------------------------------------------


class Weatherman(App):
    CSS = """
    Screen { background: #0d1117; }
    #left { width: 38; border-right: solid #30363d; }
    #right { padding: 1 2; }
    #filter_bar { height: 3; padding: 0 1; }
    Tree { background: #0d1117; color: #c9d1d9; }
    SummaryPanel, ChartsPanel, ABPanel { padding: 1 2; color: #c9d1d9; }
    .presenter SummaryPanel { text-align: center; }
    """

    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("r", "refresh", "refresh"),
        # space is Textual's Tree default for select_cursor; use priority=True
        # so our app-level binding wins. Also expose `b` (bookmark) as a
        # backup in case the priority override doesn't take on this Textual
        # version.
        Binding("space", "toggle_pin", "pin", priority=True),
        Binding("b", "toggle_pin", "pin (alt)"),
        Binding("c", "show_ab", "compare"),
        Binding("p", "open_png", "PNG"),
        Binding("slash", "focus_filter", "/filter"),
        Binding("f1", "toggle_presenter", "presenter"),
        Binding("1", "tab('summary')", "summary"),
        Binding("2", "tab('charts')", "charts"),
        Binding("3", "tab('raw')", "raw"),
        Binding("4", "tab('events')", "events"),
        Binding("5", "tab('ab')", "A/B"),
        Binding("6", "tab('search')", "search"),
        Binding("n", "narrate", "narrate→md"),
    ]

    presenter: reactive[bool] = reactive(False)
    filter_text: reactive[str] = reactive("")

    def __init__(self, runs_dir: Path):
        super().__init__()
        self.runs_dir = runs_dir
        self.runs: list[bench_data.RunIndex] = []
        self.pinned_paths: set[Path] = set()
        self.selected: bench_data.RunIndex | None = None
        self.highlighted: bench_data.RunIndex | None = None
        self._last_mtime: float | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            with Vertical(id="left"):
                yield Input(placeholder="filter (model / quant / rig)…", id="filter_bar")
                yield Tree("runs", id="tree")
            with Container(id="right"):
                with TabbedContent(initial="summary", id="tabs"):
                    with TabPane("Summary", id="summary"):
                        with VerticalScroll():
                            yield SummaryPanel(id="summary_panel")
                    with TabPane("Charts", id="charts"):
                        with VerticalScroll():
                            yield ChartsPanel(id="charts_panel")
                    with TabPane("Raw", id="raw"):
                        yield RawPanel(id="raw_panel")
                    with TabPane("Events", id="events"):
                        yield EventsPanel(id="events_panel")
                    with TabPane("A/B", id="ab"):
                        with VerticalScroll():
                            yield ABPanel(id="ab_panel")
                    with TabPane("Search", id="search"):
                        yield SearchPanel(id="search_panel")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"weatherman — {self.runs_dir}"
        self.refresh_runs()
        self.set_interval(5.0, self._auto_refresh)

    # ---- data ----

    def refresh_runs(self) -> None:
        self.runs = bench_data.discover_runs(self.runs_dir)
        self.rebuild_tree()

    def _auto_refresh(self) -> None:
        if not self.runs_dir.exists():
            return
        m = self.runs_dir.stat().st_mtime
        if self._last_mtime is None or m > self._last_mtime:
            self._last_mtime = m
            self.refresh_runs()

    def _filtered(self) -> list[bench_data.RunIndex]:
        if not self.filter_text:
            return self.runs
        ft = self.filter_text.lower()
        return [r for r in self.runs
                if ft in r.model.lower()
                or ft in r.quant.lower()
                or ft in r.rig_label.lower()
                or (r.draft_model and ft in r.draft_model.lower())]

    def rebuild_tree(self) -> None:
        tree: Tree = self.query_one("#tree", Tree)
        tree.clear()
        tree.root.expand()
        # group: model → quant → run
        by_model: dict[str, dict[str, list[bench_data.RunIndex]]] = {}
        for r in self._filtered():
            by_model.setdefault(r.model, {}).setdefault(r.quant, []).append(r)
        for model in sorted(by_model):
            model_node = tree.root.add(f"[b]{model}[/b]", expand=True)
            for quant in sorted(by_model[model]):
                quant_node = model_node.add(f"{quant}", expand=True)
                for r in sorted(by_model[model][quant], key=lambda x: x.timestamp, reverse=True):
                    pin = "● " if r.path in self.pinned_paths else "○ "
                    dec_trace = trace_values(r, "decode_tps")
                    sp = bench_data.spark(dec_trace, 12) if dec_trace else "            "
                    spec_hint = f"  [magenta]spec={r.spec_type}[/]" if r.spec_type else ""
                    # Health marker — surfaces runs that recorded warn/error
                    # events (VRAM spillover, fit reshape, OOM, etc.) without
                    # needing to drill in. Reads events.log on-demand; cheap.
                    warn_mark = "[red]⚠ [/]" if bench_data.run_has_warnings(r.path) else ""
                    ts_local = bench_data.format_timestamp_local(r.timestamp, with_tz=False)
                    label = (f"{pin}{warn_mark}{ts_local}  {r.rig_label}  "
                             f"[yellow]{sp}[/]  "
                             f"[dim]dec={fmt_num(r.decode_tps_mean, ' t/s', 6)}[/dim]"
                             f"{spec_hint}")
                    quant_node.add_leaf(label, data=r)

    # ---- events ----

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        # Enter pressed — explicit select, also updates the cursor target.
        data = event.node.data
        if isinstance(data, bench_data.RunIndex):
            self.selected = data
            self.highlighted = data
            self._refresh_detail()

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        # Cursor moved to a node — track for pin/PNG actions and update detail
        # so the right pane mirrors what the cursor is on (no Enter required).
        data = event.node.data
        if isinstance(data, bench_data.RunIndex):
            self.highlighted = data
            self.selected = data
            self._refresh_detail()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter_bar":
            self.filter_text = event.value
            self.rebuild_tree()
        elif event.input.id == "search_input":
            self.query_one("#search_panel", SearchPanel).update_for_query(
                self.runs_dir, event.value,
            )

    def _refresh_detail(self) -> None:
        size = self.size
        w, h = size.width - 42, size.height - 6  # rough right-pane size
        self.query_one("#summary_panel", SummaryPanel).update_run(self.selected, self.presenter)
        self.query_one("#charts_panel", ChartsPanel).update_run(self.selected, w, h)
        self.query_one("#raw_panel", RawPanel).update_run(self.selected)
        self.query_one("#events_panel", EventsPanel).update_run(self.selected)
        pinned = [r for r in self.runs if r.path in self.pinned_paths]
        self.query_one("#ab_panel", ABPanel).update_pinned(pinned, w, h)

    def watch_presenter(self, _old: bool, new: bool) -> None:
        if new:
            self.add_class("presenter")
        else:
            self.remove_class("presenter")
        self._refresh_detail()

    # ---- actions ----

    def action_refresh(self) -> None:
        self.refresh_runs()

    def action_toggle_pin(self) -> None:
        target = self.highlighted or self.selected
        if target is None:
            # If you see THIS message, the binding IS firing but no node
            # is highlighted yet (cursor isn't on a leaf). If you see no
            # message at all, the keybinding isn't reaching this handler
            # at all — likely Tree widget is intercepting the key.
            self.notify("pin: cursor not on a run leaf yet", severity="warning")
            return
        if target.path in self.pinned_paths:
            self.pinned_paths.remove(target.path)
            self.notify(f"unpinned {target.path.name}")
        else:
            self.pinned_paths.add(target.path)
            self.notify(f"pinned {target.path.name}  ({len(self.pinned_paths)} pinned)")
        self.rebuild_tree()
        self._refresh_detail()

    def action_show_ab(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "ab"
        self._refresh_detail()

    def action_open_png(self) -> None:
        if self.selected is None:
            return
        plots = self.selected.path / "plots"
        # Try the most useful chart first, then fall back. Multi-preset runs
        # write overlay + per-preset facets; single-preset runs write the flat
        # filename.
        candidates: list[Path] = [
            plots / "decode_vs_gen_by_preset.png",       # multi-preset overlay
            plots / "decode_vs_gen.png",                 # single-preset
        ]
        candidates += sorted(plots.glob("decode_vs_gen__*.png"))
        png = next((p for p in candidates if p.exists()), None)
        if png is None:
            self.notify("no decode plot found in plots/", severity="warning")
            self.bell()
            return
        try:
            subprocess.Popen(["xdg-open", str(png)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.notify(f"opened {png.name}")
        except FileNotFoundError:
            self.notify("xdg-open not found", severity="error")
            self.bell()

    def action_focus_filter(self) -> None:
        self.query_one("#filter_bar", Input).focus()

    def action_narrate(self) -> None:
        """Export the highlighted run's events.log as a markdown
        narrative (#19) — writes narrative.md to the run dir for
        embedding in blog posts, video scripts, or shared write-ups.
        Bell + skip silently if no run is highlighted."""
        target = self.highlighted or self.selected
        if target is None:
            self.bell()
            return
        narrative = bench_data.narrate_run_events(target.path)
        if not narrative:
            self.bell()
            return
        out = target.path / "narrative.md"
        out.write_text(narrative)
        # Brief notification via the title bar (Textual doesn't expose a
        # clean toast API on all versions; title-flash is reliably visible).
        self.title = f"weatherman — wrote {out.name} to {target.path.name}"

    def action_toggle_presenter(self) -> None:
        self.presenter = not self.presenter

    def action_tab(self, name: str) -> None:
        self.query_one("#tabs", TabbedContent).active = name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", nargs="?", type=Path, default=Path("runs"))
    args = ap.parse_args()
    if not args.runs_dir.exists():
        print(f"runs dir not found: {args.runs_dir}", file=sys.stderr)
        return 1
    Weatherman(args.runs_dir).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
