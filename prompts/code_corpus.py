"""prompts/code_corpus.py — katabasis benchmark corpus (real Python source).

Concatenation of the katabasis project source files. Used by the 'code' prompt
preset to elicit code-style decoding (high speculative-decoding acceptance).

NOTE: this file is read-only test data. Do not import or run.
"""

# ======================================================================
# kata.py
# ======================================================================

#!/usr/bin/env python3
"""katabasis — llama.cpp benchmark harness with live dashboard.

Usage:
    python kata.py bench.yaml
    python kata.py bench.yaml --manifest-only   # write system.json and exit
    python kata.py bench.yaml --no-live         # plain progress, no rich dash
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import requests
import yaml
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

import bench_data
import sysinfo


# ---------- config ----------------------------------------------------------


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_run_dir(cfg: dict) -> Path:
    base = Path((cfg.get("output") or {}).get("dir", "runs"))
    rig = (cfg.get("hardware") or {}).get("rig_label", "rig")
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = base / f"{ts}__{rig}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------- server ----------------------------------------------------------


def assemble_server_argv(cfg: dict) -> tuple[list[str], dict[str, str], dict]:
    """Return (argv, env_overrides, notes). Applies hardware.* to args/env, and
    auto-bumps --parallel to match max(concurrency_levels) when needed."""
    server = cfg.get("server", {}) or {}
    hw = cfg.get("hardware", {}) or {}
    sweep = cfg.get("sweep", {}) or {}
    binary = server.get("binary") or "llama-server"
    args = list(server.get("args") or [])
    notes: dict[str, str] = {}

    flat = bench_data.server_args_to_dict(args)

    # Hardware passthrough — only added when not already in user args.
    if hw.get("tensor_split") and "--tensor-split" not in flat:
        args.append({"--tensor-split": hw["tensor_split"]})
    if hw.get("main_gpu") is not None and "--main-gpu" not in flat:
        args.append({"--main-gpu": hw["main_gpu"]})

    # Auto-bump --parallel to max(concurrency_levels). Never reduce.
    concurrency_levels = bench_data.get_concurrency_levels(sweep)
    required_parallel = max(concurrency_levels)
    if required_parallel > 1:
        try:
            existing_parallel = int(flat.get("--parallel", flat.get("-np", 1)))
        except (TypeError, ValueError):
            existing_parallel = 1
        if existing_parallel < required_parallel:
            # Override / inject --parallel. Remove any existing --parallel / -np
            # entries first, then append the new value.
            args = [e for e in args
                    if not (isinstance(e, dict) and set(e.keys()) & {"--parallel", "-np"})
                    and e not in ("--parallel", "-np")]
            args.append({"--parallel": required_parallel})
            notes["parallel_override"] = (
                f"--parallel auto-bumped from {existing_parallel} to "
                f"{required_parallel} to match max(concurrency_levels)"
            )

    argv = [binary] + bench_data.server_args_to_argv(args)

    env_overrides = {}
    cvd = hw.get("cuda_visible_devices")
    if cvd is not None:
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(cvd)
        env_overrides["HIP_VISIBLE_DEVICES"] = str(cvd)
    return argv, env_overrides, notes


def get_slots_state(endpoint: str, timeout: float = 5.0) -> list[dict] | None:
    """Fetch /slots. Returns list of slot dicts or None if unavailable."""
    url = endpoint.rstrip("/") + "/slots"
    try:
        r = requests.get(url, timeout=timeout)
        if not r.ok:
            return None
        data = r.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "slots" in data:
            return data["slots"]
        return None
    except requests.RequestException:
        return None


def _slot_is_busy(slot: dict) -> bool:
    """Robust busy check — llama-server's /slots schema has shifted over time."""
    if slot.get("is_processing"):
        return True
    # Newer: id_task = -1 means idle, anything else means a task is bound.
    id_task = slot.get("id_task")
    if isinstance(id_task, int) and id_task != -1:
        return True
    # Older: integer state, 0 = idle.
    state = slot.get("state")
    if isinstance(state, int) and state != 0:
        return True
    # Sometimes state is a string label.
    if isinstance(state, str) and state.lower() not in ("idle", ""):
        return True
    return False


def peek_busy_slots(endpoint: str) -> int:
    """Return current count of busy slots, 0 if /slots unavailable."""
    slots = get_slots_state(endpoint)
    if slots is None:
        return 0
    return sum(1 for s in slots if _slot_is_busy(s))


def wait_for_server_idle(endpoint: str, timeout: float = 30.0,
                          poll_interval: float = 0.2) -> bool:
    """Poll /slots until all slots report non-processing. False on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if peek_busy_slots(endpoint) == 0:
            return True
        time.sleep(poll_interval)
    return False


class ConcurrencySampler:
    """Background thread polling /slots during a batch, recording max-observed
    busy-slot count. Used to verify that the server's actual parallelism
    matches what the client requested for the batch."""

    def __init__(self, endpoint: str, interval: float = 0.1):
        self.endpoint = endpoint
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.max_busy = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                busy = peek_busy_slots(self.endpoint)
                if busy > self.max_busy:
                    self.max_busy = busy
            except Exception:
                pass


def wait_for_health(endpoint: str, timeout_s: float, console: Console) -> None:
    deadline = time.time() + timeout_s
    url = endpoint.rstrip("/") + "/health"
    last_err: str | None = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2.0)
            if r.status_code == 200:
                return
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(1.0)
    raise RuntimeError(f"llama-server /health timeout after {timeout_s}s ({last_err})")


def launch_server(argv: list[str], env_overrides: dict[str, str],
                  log_path: Path, console: Console) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(env_overrides)
    console.print(Panel(
        Text(" ".join(shlex.quote(a) for a in argv), style="bright_white"),
        title="[bold]launching llama-server[/bold]",
        subtitle=", ".join(f"{k}={v}" for k, v in env_overrides.items()) or None,
        border_style="cyan",
    ))
    log_f = log_path.open("w")
    proc = subprocess.Popen(
        argv,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid if os.name == "posix" else None,
    )
    return proc


def terminate_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
    except (ProcessLookupError, OSError):
        pass


# ---------- prompts ---------------------------------------------------------


def tokenize(endpoint: str, text: str) -> list[int]:
    """Round-trip text through the server's /tokenize endpoint."""
    if not text:
        return []
    url = endpoint.rstrip("/") + "/tokenize"
    r = requests.post(url, json={"content": text}, timeout=60)
    r.raise_for_status()
    return r.json().get("tokens", [])


def detokenize(endpoint: str, tokens: list[int]) -> str:
    """Round-trip tokens back to text via /detokenize."""
    if not tokens:
        return ""
    url = endpoint.rstrip("/") + "/detokenize"
    r = requests.post(url, json={"tokens": tokens}, timeout=60)
    r.raise_for_status()
    return r.json().get("content", "")


# ---------- single request --------------------------------------------------


def run_completion(endpoint: str, prompt: str, n_predict: int,
                   on_token: callable | None = None) -> dict:
    """POST to native /completion (streaming) and return the final timings/metadata.

    Returns dict with: prefill_tps, decode_tps, prompt_n, predicted_n,
                       draft_n, draft_accepted, raw_timings, text
    """
    url = endpoint.rstrip("/") + "/completion"
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "cache_prompt": False,
        "stream": True,
        "temperature": 0.0,
        "top_k": 1,  # greedy for determinism across rounds
    }
    text_chunks: list[str] = []
    final: dict[str, Any] = {}
    t_start = time.time()
    t_first_token: float | None = None
    with requests.post(url, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            # llama-server streams Server-Sent-Events style: "data: {...}"
            if raw_line.startswith("data:"):
                raw_line = raw_line[5:].strip()
            if raw_line == "[DONE]":
                continue
            try:
                evt = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            content = evt.get("content")
            if content:
                if t_first_token is None:
                    t_first_token = time.time()
                text_chunks.append(content)
                if on_token:
                    on_token(content)
            if evt.get("stop") or evt.get("stopped_eos") or evt.get("stopped_limit") \
                    or evt.get("stopped_word"):
                final = evt
    wall = time.time() - t_start
    ttft_ms = ((t_first_token - t_start) * 1000.0) if t_first_token is not None else None

    timings = final.get("timings") or {}
    prompt_n = timings.get("prompt_n") or final.get("tokens_evaluated")
    predicted_n = timings.get("predicted_n") or final.get("tokens_predicted")
    # llama.cpp's preferred fields are predicted_per_second / prompt_per_second
    prefill_tps = timings.get("prompt_per_second")
    decode_tps = timings.get("predicted_per_second")
    # Fall back to per_token_ms if needed
    if prefill_tps is None and timings.get("prompt_per_token_ms"):
        prefill_tps = 1000.0 / timings["prompt_per_token_ms"]
    if decode_tps is None and timings.get("predicted_per_token_ms"):
        decode_tps = 1000.0 / timings["predicted_per_token_ms"]

    # Draft / speculative fields — naming has shifted across llama.cpp versions
    draft_n = (timings.get("draft_n") or timings.get("n_drafted")
               or final.get("draft_n") or final.get("n_drafted"))
    draft_accepted = (timings.get("draft_n_accepted") or timings.get("n_accept")
                      or final.get("draft_n_accepted") or final.get("n_accept"))

    return {
        "prompt_n": prompt_n,
        "predicted_n": predicted_n,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "ttft_ms": ttft_ms,
        "draft_n": draft_n,
        "draft_accepted": draft_accepted,
        "wall_s": wall,
        "raw_timings": timings,
        "text": "".join(text_chunks),
    }


# ---------- GPU telemetry ---------------------------------------------------


class GpuTelemetry:
    HISTORY_LEN = 120  # samples (≈ 2 min at 1 Hz)

    def __init__(self, out_path: Path):
        self.out_path = out_path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.latest: dict[int, dict] = {}
        self.history: dict[int, deque[dict]] = {}
        # Per-GPU static info captured on first sample: total VRAM, power limit, name.
        self.static: dict[int, dict] = {}

    def start(self) -> None:
        if not _has_cmd("nvidia-smi"):
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,"
            "temperature.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits",
            "-l", "1",
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    text=True)
        except FileNotFoundError:
            return
        with self.out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "index", "util_gpu", "util_mem",
                        "temp_c", "power_w", "vram_used_mib"])
            while not self._stop.is_set():
                line = proc.stdout.readline() if proc.stdout else ""
                if not line:
                    time.sleep(0.1)
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 7:
                    continue
                w.writerow(parts)
                f.flush()
                try:
                    idx = int(parts[1])
                    sample = {
                        "util_gpu": float(parts[2]) if parts[2].replace(".", "").isdigit() else None,
                        "temp_c": float(parts[4]) if parts[4].replace(".", "").isdigit() else None,
                        "power_w": float(parts[5]) if parts[5].replace(".", "").isdigit() else None,
                        "vram_used_mib": int(float(parts[6])) if parts[6].replace(".", "").isdigit() else None,
                    }
                    self.latest[idx] = sample
                    h = self.history.setdefault(idx, deque(maxlen=self.HISTORY_LEN))
                    h.append(sample)
                except (ValueError, IndexError):
                    pass
        proc.terminate()


def _has_cmd(name: str) -> bool:
    import shutil
    return shutil.which(name) is not None


# ---------- live dashboard --------------------------------------------------


_spark = bench_data.spark
_braille_bars = bench_data.braille_bars


class Dashboard:
    """rich Live layout, nvtop-style: per-GPU sparklines + bench sparklines + tokens."""

    SPARK_W = 50  # sparkline width in cells
    GPU_CHART_H = 3  # rows of Braille per metric (per-GPU charts)
    BENCH_CHART_H = 3  # rows of Braille for the bench panel charts

    def __init__(self, console: Console, total: int, system: dict, cfg: dict,
                 telemetry: GpuTelemetry | None):
        self.console = console
        self.total = total
        self.done = 0
        self.system = system
        self.cfg = cfg
        self.telemetry = telemetry
        self.current: dict | None = None
        self.recent_decode: deque[float] = deque(maxlen=self.SPARK_W)
        self.recent_prefill: deque[float] = deque(maxlen=self.SPARK_W)
        self.recent_ttft: deque[float] = deque(maxlen=self.SPARK_W)
        self.recent_accept: deque[float] = deque(maxlen=self.SPARK_W)
        self.token_buf: deque[str] = deque(maxlen=400)

        # Static per-GPU info (name, vram total, power limit) from the manifest,
        # keyed by the CUDA index the *server* sees (post-CUDA_VISIBLE_DEVICES).
        # The telemetry thread keys by the OS index, which matches manifest.gpus.
        self._gpu_meta: dict[int, dict] = {}
        for g in (system.get("gpus") or []):
            self._gpu_meta[int(g.get("index", -1))] = {
                "name": _short_gpu_name(g.get("name")),
                "vram_total_mib": g.get("vram_total_mib") or 0,
                "power_limit_w": g.get("power_limit_w") or 0,
            }
        # Selection set — only render GPUs the server can see.
        sel = (system.get("selection") or {}).get("cuda_visible_devices")
        self._selected: set[int] | None = None
        if sel is not None:
            self._selected = {int(s) for s in str(sel).split(",") if s.strip().isdigit()}

        self._ticker_stop = threading.Event()
        self._ticker: threading.Thread | None = None
        self.layout = self._build_layout()

    # ---- layout ------------------------------------------------------------

    def _gpu_panel_height(self) -> int:
        n_gpu = sum(1 for i in self._gpu_meta
                    if self._selected is None or i in self._selected)
        # title border (2) + per GPU: 1 header + 4 metrics × GPU_CHART_H
        per_gpu = 1 + 4 * self.GPU_CHART_H
        return max(7, 2 + n_gpu * per_gpu)

    def _bench_panel_height(self) -> int:
        # title border (2) + prefill + decode + ttft + optional accept,
        # each metric using BENCH_CHART_H rows.
        n_metrics = 2 + (1 if self.recent_ttft else 0) + (1 if self.recent_accept else 0)
        return 2 + n_metrics * self.BENCH_CHART_H

    def _build_layout(self) -> Layout:
        root = Layout()
        root.split_column(
            Layout(name="header", size=3),
            Layout(name="rig", size=5),
            Layout(name="gpu", size=self._gpu_panel_height()),
            Layout(name="bench", size=self._bench_panel_height()),
            Layout(name="stream", size=5),
        )
        return root

    # ---- panels ------------------------------------------------------------

    def _header(self) -> Panel:
        name = self.cfg.get("name", "benchmark")
        pct = (self.done / self.total * 100.0) if self.total else 0.0
        cur = self.current or {}
        cur_str = ((f"{cur.get('preset','?')}  ctx={cur.get('context_size')} "
                    f"g={cur.get('gen_size')}  r={cur.get('round')}"
                    + (f"  c={cur.get('concurrency')}"
                       if cur.get('concurrency', 1) > 1 else ""))
                   if self.current else "idle")
        bar_filled = int(pct / 100.0 * 30)
        bar = "█" * bar_filled + "░" * (30 - bar_filled)
        text = Text.assemble(
            (f"{name}  ", "bold cyan"),
            (f"[{bar}] ", "bright_blue"),
            (f"{self.done}/{self.total}  ", "bright_white"),
            (f"now: {cur_str}", "dim"),
        )
        return Panel(text, border_style="cyan", padding=(0, 1))

    def _rig_panel(self) -> Panel:
        m = self.system or {}
        host = m.get("host") or {}
        cpu = m.get("cpu") or {}
        mem = m.get("memory") or {}
        llc = m.get("llama_cpp") or {}
        sel = m.get("selection") or {}

        flat = bench_data.server_args_to_dict((self.cfg.get("server") or {}).get("args") or [])
        model = (flat.get("--hf-repo") or flat.get("-hf")
                 or flat.get("--model") or flat.get("-m") or "?")
        draft = flat.get("--model-draft") or flat.get("-md")
        # Compact headline flags
        headline = []
        for k in ("--batch-size", "--ubatch-size", "--ctx-size", "--parallel",
                  "-fa", "--cache-type-k", "--cache-type-v", "--spec-type"):
            if k in flat:
                v = flat[k]
                headline.append(k.lstrip("-") if v is True else f"{k.lstrip('-')}={v}")
        flags_str = "  ".join(headline) or "(no flags)"

        g = Table.grid(padding=(0, 1))
        g.add_column(style="bold magenta", no_wrap=True)
        g.add_column(overflow="fold")
        g.add_row(
            "rig",
            f"{m.get('rig_label','?')}  "
            f"[dim]{host.get('hostname','?')}[/dim]  "
            f"{(cpu.get('model') or '?')}  "
            f"{cpu.get('cores_physical','?')}c/{cpu.get('cores_logical','?')}t  "
            f"{mem.get('total_gb','?')}G  "
            f"[dim]llama.cpp {(llc.get('git_commit') or '?')[:10]}[/dim]  "
            f"[dim]cvd={sel.get('cuda_visible_devices')}[/dim]",
        )
        g.add_row("model", str(model)
                  + (f"  [dim]draft={draft}[/dim]" if draft else ""))
        g.add_row("flags", flags_str)
        return Panel(g, title="rig + model", border_style="magenta",
                     padding=(0, 1))

    def _metric_block(self, label: str, values, lo: float | None, hi: float | None,
                      cur: str, peak: str, style: str, height: int) -> list[Text]:
        """Render one metric as height-row Braille chart with label/values on
        the first row, indent on remaining rows."""
        bars = _braille_bars(values, self.SPARK_W, height, lo=lo, hi=hi)
        out: list[Text] = []
        # Row 0: label + chart + cur + peak.
        out.append(Text.assemble(
            (f"{label:>5} ", "dim"),
            (bars[0], style),
            (f"  cur {cur}", "bright_white"),
            (f"  peak {peak}", "dim"),
        ))
        # Subsequent rows: just chart, indented to match label width.
        for row in bars[1:]:
            out.append(Text.assemble(("      ", "dim"), (row, style)))
        return out

    def _gpu_panel(self) -> Panel:
        if not self.telemetry:
            return Panel(Text("(gpu telemetry disabled)", style="dim"),
                         title="gpu", border_style="dim")
        rows: list[Text] = []
        indices = sorted(self._gpu_meta) if self._gpu_meta else sorted(self.telemetry.history)
        for idx in indices:
            if self._selected is not None and idx not in self._selected:
                continue
            meta = self._gpu_meta.get(idx, {})
            hist = list(self.telemetry.history.get(idx, []))
            name = meta.get("name") or f"gpu{idx}"
            vram_total = meta.get("vram_total_mib") or 0
            pwr_limit = meta.get("power_limit_w") or 0

            def col(key: str) -> list[float]:
                return [s.get(key) for s in hist if s.get(key) is not None]

            util = col("util_gpu")
            pwr = col("power_w")
            vram = col("vram_used_mib")
            temp = col("temp_c")

            cur_u = f"{util[-1]:3.0f}%" if util else "  -%"
            pk_u = f"{max(util):3.0f}%" if util else "  -%"
            cur_p = f"{pwr[-1]:4.0f}W" if pwr else "   -W"
            pk_p = f"{max(pwr):4.0f}W" if pwr else "   -W"
            cur_v = f"{vram[-1]/1024:4.1f}G" if vram else "   -G"
            pk_v = f"{max(vram)/1024:4.1f}G" if vram else "   -G"
            cur_t = f"{temp[-1]:3.0f}°C" if temp else "  -°C"
            pk_t = f"{max(temp):3.0f}°C" if temp else "  -°C"

            head = Text.assemble(
                (f"GPU {idx}  ", "bold magenta"),
                (name, "bold"),
                (f"   vram cap {vram_total/1024:.0f}G   pwr cap {pwr_limit:.0f}W",
                 "dim"),
            )
            rows.append(head)
            h = self.GPU_CHART_H
            rows += self._metric_block("util", util, 0, 100, cur_u, pk_u, "bright_green", h)
            rows += self._metric_block("pwr", pwr, 0, pwr_limit or None, cur_p, pk_p, "bright_yellow", h)
            rows += self._metric_block("vram", vram, 0, vram_total or None, cur_v, pk_v, "bright_cyan", h)
            rows += self._metric_block("temp", temp, 30, 90, cur_t, pk_t, "bright_red", h)
        if not rows:
            rows = [Text("(no GPU samples yet)", style="dim")]
        return Panel(Group(*rows), title="gpu", border_style="magenta",
                     padding=(0, 1))

    def _bench_panel(self) -> Panel:
        from statistics import mean

        def block(label: str, vals, unit: str, style: str,
                  lo: float | None, hi: float | None,
                  fmt: str = "{:7.1f}", extra: str | None = None) -> list[Text]:
            cur = fmt.format(vals[-1]) if vals else "    -- "
            mn = fmt.format(mean(vals)) if vals else "    -- "
            bars = _braille_bars(vals, self.SPARK_W, self.BENCH_CHART_H, lo=lo, hi=hi)
            head_segs = [
                (f"{label:>7} ", "dim"),
                (bars[0], style),
                (f" {cur}{unit}", "bright_white"),
            ]
            if extra:
                head_segs.append((f" {extra}", "bright_cyan"))
            head_segs.append((f" μ {mn}{unit}", "dim"))
            out = [Text.assemble(*head_segs)]
            for row in bars[1:]:
                out.append(Text.assemble(("        ", "dim"), (row, style)))
            return out

        # Inline ms/tok on the decode line — same data as decode t/s but in
        # latency form, which is often easier to read at a glance.
        ms_per_tok_cur = (
            f"{1000.0/self.recent_decode[-1]:5.2f} ms/t"
            if self.recent_decode and self.recent_decode[-1] > 0
            else None
        )

        rows: list[Text] = []
        rows += block("prefill", self.recent_prefill, "t/s", "bright_green", 0, None)
        rows += block("decode", self.recent_decode, "t/s", "bright_yellow", 0, None,
                      extra=ms_per_tok_cur)
        if self.recent_ttft:
            rows += block("TTFT", self.recent_ttft, "ms", "bright_blue", 0, None,
                          fmt="{:7.0f}")
        if self.recent_accept:
            rows += block("accept",
                          [v * 100 for v in self.recent_accept],
                          "%", "bright_magenta", 0, 100, "{:6.1f}")
        return Panel(Group(*rows), title="benchmark", border_style="yellow",
                     padding=(0, 1))

    def _stream(self) -> Panel:
        # Panel is 5 rows total → ~3 rows of content. Trim aggressively so
        # the panel never wraps and pushes itself out of bounds.
        text = "".join(self.token_buf).replace("\n", " ")
        text = text[-360:].lstrip()
        return Panel(Text(text, style="bright_white", overflow="ellipsis", no_wrap=False),
                     title="tokens", border_style="green",
                     padding=(0, 1), height=5)

    # ---- update + ticker ---------------------------------------------------

    def update(self, current: dict | None = None, row: dict | None = None,
               token: str | None = None) -> None:
        if current is not None:
            self.current = current
        if row is not None:
            self.done += 1
            if row.get("decode_tps"):
                self.recent_decode.append(row["decode_tps"])
            if row.get("prefill_tps"):
                self.recent_prefill.append(row["prefill_tps"])
            if row.get("ttft_ms") is not None:
                self.recent_ttft.append(row["ttft_ms"])
            if row.get("draft_n") and row.get("draft_accepted") is not None and row["draft_n"]:
                self.recent_accept.append(row["draft_accepted"] / row["draft_n"])
        if token:
            self.token_buf.append(token)

        self.layout["header"].update(self._header())
        self.layout["rig"].update(self._rig_panel())
        self.layout["gpu"].update(self._gpu_panel())
        self.layout["bench"].update(self._bench_panel())
        self.layout["stream"].update(self._stream())

    def start_ticker(self, interval_s: float = 0.25) -> None:
        def _run():
            while not self._ticker_stop.wait(interval_s):
                try:
                    self.update()
                except Exception:
                    pass
        self._ticker = threading.Thread(target=_run, daemon=True)
        self._ticker.start()

    def stop_ticker(self) -> None:
        self._ticker_stop.set()
        if self._ticker:
            self._ticker.join(timeout=1.0)


# ---------- summary --------------------------------------------------------


def write_summary(run_dir: Path) -> None:
    """Headline numbers for summary.json — token-weighted across the whole run."""
    rows = bench_data.load_jsonl(run_dir / "raw.jsonl")
    if not rows:
        return
    from statistics import mean
    # Token-weighted overall rates.
    pred_n = [r.get("predicted_n") or 0 for r in rows if (r.get("decode_tps") or 0) > 0]
    pred_t = [(r["predicted_n"] / r["decode_tps"]) for r in rows
              if (r.get("decode_tps") or 0) > 0 and r.get("predicted_n")]
    prompt_n = [r.get("prompt_n") or 0 for r in rows if (r.get("prefill_tps") or 0) > 0]
    prompt_t = [(r["prompt_n"] / r["prefill_tps"]) for r in rows
                if (r.get("prefill_tps") or 0) > 0 and r.get("prompt_n")]
    decode_tw = (sum(pred_n) / sum(pred_t)) if (pred_t and sum(pred_t) > 0) else None
    prefill_tw = (sum(prompt_n) / sum(prompt_t)) if (prompt_t and sum(prompt_t) > 0) else None
    accept_vals = [
        r["draft_accepted"] / r["draft_n"]
        for r in rows if r.get("draft_n") and r.get("draft_accepted") is not None
    ]
    ttft_vals = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
    summary = {
        "decode_tps_mean": decode_tw,
        "decode_ms_per_token": (1000.0 / decode_tw) if decode_tw else None,
        "prefill_tps_mean": prefill_tw,
        "ttft_ms_mean": mean(ttft_vals) if ttft_vals else None,
        "accept_rate_mean": mean(accept_vals) if accept_vals else None,
        "n_rows": len(rows),
        "_note": "rates are token-weighted (total_tokens / total_time); ttft is client-side first-chunk latency",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


# ---------- main ----------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--manifest-only", action="store_true",
                    help="capture system.json and exit")
    ap.add_argument("--no-live", action="store_true",
                    help="disable rich live dashboard (plain progress)")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip auto-plot at end")
    args = ap.parse_args()

    console = Console()
    cfg = load_config(args.config)
    run_dir = resolve_run_dir(cfg)
    console.print(f"[bold cyan]run dir:[/bold cyan] {run_dir}")

    # 1. config echo (compact)
    _render_config_panel(console, cfg)

    # 2. server argv + manifest capture
    argv, env_overrides, server_notes = assemble_server_argv(cfg)
    if server_notes.get("parallel_override"):
        console.print(f"[yellow]{server_notes['parallel_override']}[/yellow]")
    hw = cfg.get("hardware", {}) or {}
    manifest = sysinfo.capture(
        rig_label=hw.get("rig_label", "rig"),
        selection={
            "cuda_visible_devices": hw.get("cuda_visible_devices"),
            "tensor_split": hw.get("tensor_split"),
            "main_gpu": hw.get("main_gpu"),
        },
        config_snapshot={
            "name": cfg.get("name"),
            "server_args": bench_data.server_args_to_dict(cfg.get("server", {}).get("args")),
            "sweep": cfg.get("sweep", {}),
            "argv": argv,
            "env": env_overrides,
        },
        binary=(cfg.get("server", {}) or {}).get("binary"),
    )
    sysinfo.write(manifest, run_dir)
    _render_manifest_panel(console, manifest)

    if args.manifest_only:
        console.print(f"[green]manifest-only mode: wrote {run_dir / 'system.json'}[/green]")
        return 0

    # 3. launch server (if requested)
    server_cfg = cfg.get("server", {}) or {}
    endpoint = server_cfg.get("endpoint", "http://localhost:8080")
    proc: subprocess.Popen | None = None
    try:
        if server_cfg.get("launch"):
            proc = launch_server(argv, env_overrides, run_dir / "server.log", console)
            with console.status("[bold cyan]waiting for /health…[/bold cyan]"):
                wait_for_health(endpoint, server_cfg.get("health_timeout_s", 120), console)
            console.print("[green]server ready[/green]")
        else:
            wait_for_health(endpoint, 5.0, console)
            # When attached to an existing server, confirm its --parallel
            # slot count is enough for the requested concurrency_levels.
            required_parallel = max(bench_data.get_concurrency_levels(
                cfg.get("sweep", {}) or {}))
            if required_parallel > 1:
                try:
                    r = requests.get(endpoint.rstrip("/") + "/props", timeout=5)
                    if r.ok:
                        props = r.json()
                        # llama-server names it variously across versions
                        n_par = (props.get("n_parallel")
                                 or props.get("default_generation_settings", {}).get("n_parallel"))
                        if n_par is not None and int(n_par) < required_parallel:
                            console.print(
                                f"[yellow]warning: attached server reports n_parallel={n_par} "
                                f"but sweep needs {required_parallel}. Requests above the "
                                f"slot count will queue — TTFT and aggregate throughput will "
                                f"be misleading. Restart llama-server with --parallel "
                                f"{required_parallel}.[/yellow]"
                            )
                except requests.RequestException:
                    pass  # /props not available — skip the check silently

        # GPU telemetry (optional)
        telemetry: GpuTelemetry | None = None
        if (cfg.get("output") or {}).get("gpu_monitor"):
            telemetry = GpuTelemetry(run_dir / "gpu_telemetry.csv")
            telemetry.start()

        # 4. build prompts — one per (preset, context_size).
        sweep = cfg.get("sweep", {}) or {}
        context_sizes: list[int] = bench_data.get_context_sizes(sweep) or [128]
        gen_sizes: list[int] = sweep.get("gen_sizes") or [64]
        rounds: int = sweep.get("rounds", 1)
        warmup: int = sweep.get("warmup_rounds", 0)
        concurrency_levels = bench_data.get_concurrency_levels(sweep)
        prompt_presets = bench_data.get_prompt_presets(sweep)
        if not prompt_presets:
            prompt_presets = ["niah"]

        # Tokenize/detokenize closures bound to the server endpoint.
        _tok = lambda text: tokenize(endpoint, text)
        _detok = lambda toks: detokenize(endpoint, toks)

        console.print(f"[dim]building prompts: {len(prompt_presets)} preset(s) × "
                      f"{len(context_sizes)} ctx size(s) = "
                      f"{len(prompt_presets)*len(context_sizes)} unique prompts…[/dim]")
        # sized_prompts[(preset_name, ctx)] = prompt_text
        sized_prompts: dict[tuple[str, int], str] = {}
        # preset_names is the resolved-name list (file:foo.txt etc.) parallel to prompt_presets.
        preset_names: list[str] = []
        for preset_spec in prompt_presets:
            for cs in context_sizes:
                name, prompt_text = bench_data.build_prompt(
                    preset_spec, cs, tokenize_fn=_tok, detokenize_fn=_detok,
                )
                sized_prompts[(name, cs)] = prompt_text
            preset_names.append(name)

        # 5. warmup — one warmup pass per preset at smallest (ctx, gen) for that preset.
        if warmup > 0:
            console.print(f"[dim]warmup: {warmup} round(s) per preset at smallest ctx/gen…[/dim]")
            for name in preset_names:
                for _ in range(warmup):
                    try:
                        run_completion(endpoint, sized_prompts[(name, context_sizes[0])], gen_sizes[0])
                    except Exception as e:
                        console.print(f"[red]warmup error ({name}): {e}[/red]")

        # 6. sweep — round × preset × ctx × gen × concurrency
        # Each (round, preset, ctx, gen, N) is one "batch": at concurrency=N,
        # N requests fire in parallel and we record N rows tagged with the
        # same batch_id.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        total_requests = (rounds * len(preset_names) * len(context_sizes)
                          * len(gen_sizes) * sum(concurrency_levels))
        raw_path = run_dir / "raw.jsonl"
        raw_f = raw_path.open("w")
        raw_lock = threading.Lock()
        stream_tokens = (cfg.get("output") or {}).get("stream_tokens", True)
        live_mode = (cfg.get("output") or {}).get("live", True) and not args.no_live

        dashboard = Dashboard(console, total_requests, manifest, cfg, telemetry)

        def run_one_in_batch(preset: str, cs: int, gs: int,
                             concurrency: int, batch_id: str,
                             request_index: int) -> dict:
            try:
                return run_completion(
                    endpoint, sized_prompts[(preset, cs)], gs,
                    # At concurrency > 1, only the first request streams tokens
                    # to the dashboard — interleaved streams from N requests
                    # are visual noise.
                    on_token=((lambda t: dashboard.update(token=t))
                              if stream_tokens and live_mode and request_index == 0
                              else None),
                )
            except Exception as e:
                console.print(f"[red]request failed (batch {batch_id}, idx {request_index}): {e}[/red]")
                return {}

        # Quality gates & diagnostics for the concurrency sweep.
        gate_between_concurrency = bool(sweep.get("gate_between_concurrency", True))
        gate_timeout_s = float(sweep.get("gate_timeout_s", 2.0))
        # Gate only on concurrency *decrease*: c=8 → c=1 needs isolation so
        # leftover slot tail doesn't pollute the c=1 measurement. c=1 → c=8
        # is fine; the new batch saturates the GPU and subsumes any tail.
        gate_only_on_decrease = bool(sweep.get("gate_only_on_decrease", True))
        last_concurrency = {"N": None}
        empty_timings = {"count": 0, "batches_with_warnings": 0}

        # Diagnostic dump of the /slots response once at startup so we can
        # verify our busy-detection logic against this server version.
        if max(concurrency_levels) > 1:
            slots = get_slots_state(endpoint)
            if slots:
                sample = slots[0]
                kept = {k: sample.get(k) for k in
                        ("id", "id_task", "is_processing", "state")
                        if k in sample}
                console.print(f"[dim]/slots probe: {len(slots)} slot(s), "
                              f"sample fields: {kept}[/dim]")
            else:
                console.print(
                    "[yellow]/slots endpoint unavailable or returned no slots — "
                    "gating will not be able to verify server quiescence. "
                    "Ensure llama-server is started with --slots enabled.[/yellow]"
                )

        def run_batch(round_i: int, preset: str, cs: int, gs: int,
                      concurrency: int) -> None:
            batch_id = f"r{round_i}-{preset}-c{cs}-g{gs}-N{concurrency}"

            # Gate only on concurrency decrease (or any change if the user
            # opted out of gate_only_on_decrease). Short timeout — this is
            # a measurement-isolation gate, not a strict wait.
            last_N = last_concurrency["N"]
            should_gate = (
                gate_between_concurrency
                and last_N is not None
                and concurrency != last_N
                and (not gate_only_on_decrease or concurrency < last_N)
            )
            if should_gate:
                t0 = time.time()
                idle_ok = wait_for_server_idle(endpoint, timeout=gate_timeout_s)
                gate_ms = (time.time() - t0) * 1000.0
                if idle_ok:
                    console.print(
                        f"[dim]gate c{last_N}→c{concurrency}: idle in {gate_ms:.0f}ms[/]"
                    )
                else:
                    still_busy = peek_busy_slots(endpoint)
                    console.print(
                        f"[yellow]gate c{last_N}→c{concurrency}: server still has "
                        f"{still_busy} busy slot(s) after {gate_timeout_s:.1f}s; "
                        f"proceeding (measurement may be slightly polluted)[/yellow]"
                    )
            last_concurrency["N"] = concurrency

            # Pre-batch sanity check.
            pre_busy = peek_busy_slots(endpoint)
            if pre_busy > 0:
                console.print(
                    f"[yellow]warning ({batch_id}): server has {pre_busy} busy "
                    f"slot(s) before batch starts[/yellow]"
                )

            dashboard.update(current={
                "preset": preset, "context_size": cs, "gen_size": gs,
                "round": round_i, "concurrency": concurrency,
            })

            # Sampler watches /slots through the batch.
            sampler = ConcurrencySampler(endpoint, interval=0.1)
            sampler.start()

            batch_start = time.time()
            results: list[dict] = [{} for _ in range(concurrency)]
            if concurrency == 1:
                results[0] = run_one_in_batch(preset, cs, gs, concurrency, batch_id, 0)
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = {
                        pool.submit(run_one_in_batch, preset, cs, gs,
                                    concurrency, batch_id, i): i
                        for i in range(concurrency)
                    }
                    for fut in as_completed(futures):
                        i = futures[fut]
                        try:
                            results[i] = fut.result()
                        except Exception as e:
                            console.print(f"[red]batch error: {e}[/red]")
            batch_wall_s = time.time() - batch_start
            sampler.stop()
            max_observed = sampler.max_busy

            # Warn if observed concurrency exceeded requested. Shouldn't happen
            # given client-side ThreadPoolExecutor, but the check guards
            # against another client stealing the server during a sweep.
            if max_observed > concurrency:
                empty_timings["batches_with_warnings"] += 1
                console.print(
                    f"[yellow]warning ({batch_id}): peak {max_observed} busy "
                    f"slots > requested c={concurrency}[/yellow]"
                )

            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            for i, result in enumerate(results):
                if not result:
                    empty_timings["count"] += 1
                    continue
                if result.get("decode_tps") is None:
                    empty_timings["count"] += 1
                row = {
                    "ts": ts,
                    "round": round_i,
                    "preset": preset,
                    "context_size": cs,
                    "gen_size": gs,
                    "concurrency": concurrency,
                    "batch_id": batch_id,
                    "request_index": i,
                    "batch_wall_s": batch_wall_s,
                    "max_observed_concurrency": max_observed,
                    "prefill_tps": result.get("prefill_tps"),
                    "decode_tps": result.get("decode_tps"),
                    "ttft_ms": result.get("ttft_ms"),
                    "draft_n": result.get("draft_n"),
                    "draft_accepted": result.get("draft_accepted"),
                    "wall_s": result.get("wall_s"),
                    "predicted_n": result.get("predicted_n"),
                    "prompt_n": result.get("prompt_n"),
                    "raw_timings": result.get("raw_timings"),
                }
                with raw_lock:
                    raw_f.write(json.dumps(row) + "\n")
                    raw_f.flush()
                dashboard.update(row=row)

        def iter_batches():
            for r in range(1, rounds + 1):
                for preset in preset_names:
                    for cs in context_sizes:
                        for gs in gen_sizes:
                            for N in concurrency_levels:
                                yield r, preset, cs, gs, N

        if live_mode:
            with Live(dashboard.layout, console=console, refresh_per_second=8,
                      screen=False):
                dashboard.update()  # initial paint
                dashboard.start_ticker(0.25)
                try:
                    for r, preset, cs, gs, N in iter_batches():
                        run_batch(r, preset, cs, gs, N)
                finally:
                    dashboard.stop_ticker()
        else:
            with Progress(TextColumn("[progress.description]{task.description}"),
                          BarColumn(), TextColumn("{task.completed}/{task.total}"),
                          TimeElapsedColumn(), console=console) as prog:
                tid = prog.add_task("sweep", total=total_requests)
                for r, preset, cs, gs, N in iter_batches():
                    run_batch(r, preset, cs, gs, N)
                    prog.update(tid, advance=N,
                                description=f"r{r} {preset} ctx={cs} g={gs} c={N}")
        raw_f.close()
        if telemetry:
            telemetry.stop()

        # Health summary of the sweep itself.
        if empty_timings["count"] > 0:
            console.print(
                f"[yellow]diagnostic: {empty_timings['count']} request(s) "
                f"returned no decode_tps timings — dropped from aggregation. "
                f"Likely transient server churn under high concurrency.[/yellow]"
            )
        if empty_timings["batches_with_warnings"] > 0:
            console.print(
                f"[yellow]diagnostic: {empty_timings['batches_with_warnings']} "
                f"batch(es) saw more busy slots than requested concurrency.[/yellow]"
            )

        write_summary(run_dir)
        _print_summary_table(console, run_dir)

    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/yellow]")
    finally:
        if proc is not None:
            terminate_server(proc)
            console.print("[dim]server terminated[/dim]")

    # 7. auto-plot
    if not args.no_plot:
        try:
            subprocess.run([sys.executable, "plot.py", str(run_dir)], check=False)
        except Exception as e:
            console.print(f"[red]plot.py failed: {e}[/red]")

    console.print(f"\n[bold green]done.[/bold green] results in {run_dir}")
    return 0


_short_gpu_name = bench_data.short_gpu_name


def _render_config_panel(console: Console, cfg: dict) -> None:
    name = cfg.get("name") or "(unnamed)"
    server = cfg.get("server") or {}
    hw = cfg.get("hardware") or {}
    sweep = cfg.get("sweep") or {}
    output = cfg.get("output") or {}

    flat = bench_data.server_args_to_dict(server.get("args") or [])
    model = flat.get("--hf-repo") or flat.get("-hf") or flat.get("--model") or flat.get("-m") or "?"
    draft = flat.get("--model-draft") or flat.get("-md")
    spec = flat.get("--spec-type")

    # Headline flags only — full set is in system.json.
    headline_keys = ("--batch-size", "--ubatch-size", "--ctx-size", "--parallel",
                     "-fa", "--cache-type-k", "--cache-type-v", "--spec-type",
                     "--spec-draft-n-max")
    flags = " ".join(
        f"{k}={flat[k]}" if flat.get(k) not in (True, None) else k
        for k in headline_keys if k in flat
    )

    ctx = bench_data.get_context_sizes(sweep)
    gs = sweep.get("gen_sizes") or []
    rounds = sweep.get("rounds", 1)
    warmup = sweep.get("warmup_rounds", 0)
    presets = bench_data.get_prompt_presets(sweep)
    concur = bench_data.get_concurrency_levels(sweep)
    total = rounds * len(ctx) * len(gs) * max(1, len(presets)) * sum(concur)

    g = Table.grid(padding=(0, 1))
    g.add_column(style="bold cyan", no_wrap=True)
    g.add_column(overflow="fold")
    g.add_row("model", str(model) + (f"  [dim]draft={draft}[/dim]" if draft else ""))
    g.add_row("flags", flags or "[dim](none)[/dim]")
    g.add_row("sweep",
              f"ctx={ctx} × gen={gs}  rounds={rounds}  warmup={warmup}  "
              f"presets={presets}  concurrency={concur}  "
              f"[dim]({total} timed reqs)[/dim]")
    rig = hw.get("rig_label", "?")
    cvd = hw.get("cuda_visible_devices")
    g.add_row("rig", f"{rig}  cvd={cvd}  launch={bool(server.get('launch'))}  "
                     f"live={output.get('live', True)}  gpu_mon={output.get('gpu_monitor', False)}")
    console.print(Panel(g, title=f"config: [bold]{name}[/bold]",
                        border_style="cyan", padding=(0, 1)))


def _render_manifest_panel(console: Console, m: dict) -> None:
    host = m.get("host", {})
    cpu = m.get("cpu", {})
    mem = m.get("memory", {})
    llc = m.get("llama_cpp", {})
    sel = m.get("selection", {})

    head = Table.grid(padding=(0, 1))
    head.add_column(style="bold magenta", no_wrap=True)
    head.add_column(overflow="fold")
    head.add_row("rig", f"{m.get('rig_label')}  "
                       f"[dim]({host.get('hostname')} • {host.get('distro') or host.get('os')} "
                       f"k{host.get('kernel')})[/dim]")
    head.add_row("cpu/ram",
                 f"{cpu.get('model')}  {cpu.get('cores_physical')}c/{cpu.get('cores_logical')}t  "
                 f"{mem.get('total_gb')}GiB ({mem.get('available_gb')}GiB free)  "
                 f"gov={cpu.get('governor')}")
    head.add_row("llama.cpp",
                 f"commit={(llc.get('git_commit') or '?')[:10]}  "
                 f"flags=[{llc.get('build_flags') or ''}]  "
                 f"sel: cvd={sel.get('cuda_visible_devices')} "
                 f"split={sel.get('tensor_split')} main={sel.get('main_gpu')}")

    gpu_tbl = Table.grid(padding=(0, 1))
    for _ in range(7):
        gpu_tbl.add_column(no_wrap=True)
    gpu_tbl.add_row("[bold dim]idx[/]", "[bold dim]gpu[/]", "[bold dim]vram free/total[/]",
                    "[bold dim]drv[/]", "[bold dim]pwr[/]", "[bold dim]pcie[/]", "[bold dim]sel[/]")
    sel_set = set()
    if sel.get("cuda_visible_devices") is not None:
        sel_set = {s.strip() for s in str(sel["cuda_visible_devices"]).split(",") if s.strip()}
    for g in m.get("gpus", []) or []:
        in_use = "✓" if (not sel_set or str(g.get("index")) in sel_set) else " "
        vram = (f"{g.get('vram_free_mib', 0)/1024:.0f}/"
                f"{g.get('vram_total_mib', 0)/1024:.0f}G")
        gpu_tbl.add_row(
            str(g.get("index")),
            _short_gpu_name(g.get("name")),
            vram,
            str(g.get("driver") or "?"),
            f"{g.get('power_limit_w')}W",
            f"g{g.get('pcie_gen')}x{g.get('pcie_width')}",
            in_use,
        )
    console.print(Panel(Group(head, Text(""), gpu_tbl), title="system manifest",
                        border_style="magenta", padding=(0, 1)))


def _print_summary_table(console: Console, run_dir: Path) -> None:
    rows = bench_data.load_jsonl(run_dir / "raw.jsonl")
    if not rows:
        return
    presets = sorted({r.get("preset") for r in rows if r.get("preset")})
    concurrencies = sorted({r.get("concurrency", 1) for r in rows})

    # If multiple concurrency levels, lead with the workgroup-throughput
    # summary — that's the headline number for the multi-user story.
    if len(concurrencies) > 1:
        tput = bench_data.aggregate_throughput_by_concurrency(rows)
        t0 = Table(title="throughput by concurrency",
                   caption="[dim]aggregate = sum(tokens across N parallel reqs) / batch_wall_s[/dim]")
        t0.add_column("c", justify="right")
        t0.add_column("aggregate t/s", justify="right", style="bold bright_cyan")
        t0.add_column("per-req t/s", justify="right", style="yellow")
        t0.add_column("prefill t/s", justify="right")
        t0.add_column("TTFT ms (mean)", justify="right")
        t0.add_column("TTFT p95 ms", justify="right")
        t0.add_column("n batches", justify="right")
        t0.add_column("n reqs", justify="right")
        for N in concurrencies:
            m = tput.get(N, {})
            t0.add_row(
                str(N),
                f"{m['aggregate_tps_mean']:.1f}" if m.get("aggregate_tps_mean") is not None else "-",
                f"{m['per_request_tps_mean']:.1f}" if m.get("per_request_tps_mean") is not None else "-",
                f"{m['prefill_tps_mean']:.1f}" if m.get("prefill_tps_mean") is not None else "-",
                f"{m['ttft_ms_mean']:.0f}" if m.get("ttft_ms_mean") is not None else "-",
                f"{m['ttft_ms_p95']:.0f}" if m.get("ttft_ms_p95") is not None else "-",
                str(m.get("n_batches", 0)),
                str(m.get("n_requests", 0)),
            )
        console.print(t0)

    # Per-(ctx, gen) detail — one block per concurrency level when c varies.
    if len(concurrencies) == 1:
        # Single-concurrency: flat table (today's behavior).
        cells = bench_data.aggregate_by_cell(rows)
        title = "sweep summary (means)"
        if len(presets) > 1:
            title += f" — {len(presets)} presets averaged: {', '.join(presets)}"
        t = _build_cell_table(cells, title)
        console.print(t)
    else:
        # Multi-concurrency: one table per N, each headed clearly.
        cells_by_n = bench_data.aggregate_by_concurrency_cell(rows)
        for N in concurrencies:
            sub = {(c, g): v for (c, g, n), v in cells_by_n.items() if n == N}
            if not sub:
                continue
            t = _build_cell_table(sub, f"sweep detail — concurrency = {N}")
            console.print(t)

    if len(presets) > 1 or len(concurrencies) > 1:
        console.print(f"[dim]full breakdown in {run_dir}/plots/summary.md[/dim]")


def _build_cell_table(cells: dict, title: str) -> Table:
    """Render a per-(ctx, gen) cell table — token-weighted means + TTFT."""
    t = Table(title=title,
              caption="[dim]rates token-weighted (tokens / total time)[/dim]")
    t.add_column("ctx", justify="right")
    t.add_column("gen", justify="right")
    t.add_column("prefill t/s", justify="right")
    t.add_column("decode t/s", justify="right")
    t.add_column("ms/tok", justify="right")
    t.add_column("TTFT ms", justify="right")
    t.add_column("accept %", justify="right")
    t.add_column("n", justify="right")
    for (cs, gs) in sorted(cells.keys()):
        c = cells[(cs, gs)]
        t.add_row(
            str(cs), str(gs),
            f"{c['prefill_mean']:.1f}" if c.get("prefill_mean") is not None else "-",
            f"{c['decode_mean']:.1f}" if c.get("decode_mean") is not None else "-",
            f"{c['decode_ms_per_token']:.2f}" if c.get("decode_ms_per_token") is not None else "-",
            f"{c['ttft_ms_mean']:.0f}" if c.get("ttft_ms_mean") is not None else "-",
            f"{c['accept_mean']*100:.1f}" if c.get("accept_mean") is not None else "-",
            str(c.get("n", 0)),
        )
    return t


if __name__ == "__main__":
    sys.exit(main())

# ======================================================================
# bench_data.py
# ======================================================================

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
               title: str = "", x_log: bool = False) -> str:
    """Multi-series (x, y) line chart in Braille with axes + legend.

    Each series: ``{"label": str, "x": [...], "y": [...]}``.
    Returns a string with rich markup tags suitable for ``rich.console`` or
    Textual's ``Static``. Legend is rendered outside the plot area so it never
    obscures data points.
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
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1
    # 5% padding top/bottom so markers don't touch the frame.
    y_pad = (y_max - y_min) * 0.05
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


def get_concurrency_levels(sweep_cfg: dict) -> list[int]:
    """Read concurrency_levels (the list of N values to try concurrent requests).
    Always returns at least [1] so single-threaded sweeps just work."""
    levels = list(sweep_cfg.get("concurrency_levels") or [])
    if not levels:
        return [1]
    return [int(c) for c in levels if int(c) >= 1]


def aggregate_by_concurrency_cell(rows: list[dict]) -> dict[tuple[int, int, int], dict]:
    """Aggregate keyed by (context_size, gen_size, concurrency). Same metrics
    as aggregate_by_cell but with concurrency as part of the cell key. Use
    this for views that should not mix concurrency levels."""
    by_n: dict[int, list[dict]] = {}
    for r in rows:
        N = r.get("concurrency", 1)
        by_n.setdefault(N, []).append(r)
    out: dict[tuple[int, int, int], dict] = {}
    for N, sub in by_n.items():
        for k, v in aggregate_by_cell(sub).items():
            out[(k[0], k[1], N)] = v
    return out


def aggregate_throughput_by_concurrency(rows: list[dict]) -> dict[int, dict]:
    """Workgroup-throughput view: one row per concurrency level.

    Returns: {N: {
        n_batches, n_requests,
        aggregate_tps_mean,   # sum(tokens) / batch_wall_s, averaged over batches
        per_request_tps_mean, # token-weighted per-request decode t/s
        ttft_ms_mean, ttft_ms_p95,
        prefill_tps_mean,     # per-request, token-weighted
    }}
    """
    from collections import defaultdict

    # Group by batch_id first to compute per-batch aggregate.
    batches: dict[str, dict] = defaultdict(lambda: {"toks": 0, "wall": 0.0, "N": 0})
    for r in rows:
        bid = r.get("batch_id")
        if not bid:
            continue
        if r.get("predicted_n") is not None:
            batches[bid]["toks"] += r["predicted_n"]
        if r.get("batch_wall_s") is not None:
            batches[bid]["wall"] = max(batches[bid]["wall"], r["batch_wall_s"])
        batches[bid]["N"] = r.get("concurrency", 1)

    # Aggregate t/s per batch, then group by concurrency.
    agg_by_n: dict[int, list[float]] = defaultdict(list)
    for b in batches.values():
        if b["wall"] > 0 and b["toks"] > 0:
            agg_by_n[b["N"]].append(b["toks"] / b["wall"])

    # Per-request stats per concurrency.
    out: dict[int, dict] = {}
    for N in sorted({r.get("concurrency", 1) for r in rows}):
        sub = [r for r in rows if r.get("concurrency", 1) == N]
        # Token-weighted per-request decode rate.
        pred_t = [(r["predicted_n"] / r["decode_tps"]) for r in sub
                  if (r.get("decode_tps") or 0) > 0 and r.get("predicted_n")]
        pred_n = [r.get("predicted_n") or 0 for r in sub
                  if (r.get("decode_tps") or 0) > 0]
        per_req_tps = (sum(pred_n) / sum(pred_t)) if (pred_t and sum(pred_t) > 0) else None

        prompt_t = [(r["prompt_n"] / r["prefill_tps"]) for r in sub
                    if (r.get("prefill_tps") or 0) > 0 and r.get("prompt_n")]
        prompt_n = [r.get("prompt_n") or 0 for r in sub
                    if (r.get("prefill_tps") or 0) > 0]
        prefill_tps = (sum(prompt_n) / sum(prompt_t)) if (prompt_t and sum(prompt_t) > 0) else None

        ttft_vals = sorted(r["ttft_ms"] for r in sub if r.get("ttft_ms") is not None)
        ttft_mean = mean(ttft_vals) if ttft_vals else None
        ttft_p95 = ttft_vals[int(0.95 * (len(ttft_vals) - 1))] if ttft_vals else None

        n_batches = len(agg_by_n.get(N, []))
        agg_tps_mean = mean(agg_by_n[N]) if agg_by_n.get(N) else None

        out[N] = {
            "n_batches": n_batches,
            "n_requests": len(sub),
            "aggregate_tps_mean": agg_tps_mean,
            "per_request_tps_mean": per_req_tps,
            "ttft_ms_mean": ttft_mean,
            "ttft_ms_p95": ttft_p95,
            "prefill_tps_mean": prefill_tps,
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

# ======================================================================
# weatherman.py
# ======================================================================

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
        if len(idx.presets) > 1:
            parts.append("")
            parts.append(self._per_preset_block(idx))
        parts.append("")
        parts.append(self._traces_block(idx))
        return "\n".join(parts)

    def _per_concurrency_block(self, rows: list[dict],
                                concurrencies: list[int]) -> str:
        """Workgroup-throughput view: per-N aggregate t/s + per-request t/s +
        TTFT mean/p95 + request count. The headline view when concurrency
        varies."""
        tput = bench_data.aggregate_throughput_by_concurrency(rows)
        lines = ["[b]per-concurrency[/]  "
                 "[dim](aggregate = total tokens / batch wall time across N parallel reqs)[/]"]
        for N in concurrencies:
            m = tput.get(N, {})
            agg = m.get("aggregate_tps_mean")
            per = m.get("per_request_tps_mean")
            tmean = m.get("ttft_ms_mean")
            tp95 = m.get("ttft_ms_p95")
            bits = [
                f"  [b cyan]c={N:<2}[/]",
                (f"aggregate [b bright_cyan]{agg:7.1f}[/] t/s"
                 if agg is not None else "aggregate     --  t/s"),
                (f"per-req [yellow]{per:6.1f}[/] t/s"
                 if per is not None else "per-req    --  t/s"),
                (f"TTFT [blue]{tmean:5.0f}[/] ms"
                 if tmean is not None else "TTFT    -- ms"),
                (f"p95 [blue]{tp95:5.0f}[/] ms"
                 if tp95 is not None else "p95   --  ms"),
                f"[dim]({m.get('n_requests', 0)} reqs in {m.get('n_batches', 0)} batches)[/]",
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

        # ---- concurrency overlay (headline view when c varies) -----------
        if len(concurrencies) > 1:
            tput = bench_data.aggregate_throughput_by_concurrency(rows)
            parts.append("[b cyan]── concurrency: workgroup throughput ──[/]")
            agg_pts = [(N, tput[N]["aggregate_tps_mean"])
                       for N in concurrencies
                       if tput.get(N, {}).get("aggregate_tps_mean") is not None]
            per_pts = [(N, tput[N]["per_request_tps_mean"])
                       for N in concurrencies
                       if tput.get(N, {}).get("per_request_tps_mean") is not None]
            series = []
            if agg_pts:
                series.append({"label": "aggregate t/s",
                               "x": [p[0] for p in agg_pts],
                               "y": [p[1] for p in agg_pts]})
            if per_pts:
                series.append({"label": "per-request t/s",
                               "x": [p[0] for p in per_pts],
                               "y": [p[1] for p in per_pts]})
            if series:
                parts.append(bench_data.line_chart(
                    series, width=w, height=h,
                    x_label="concurrency (parallel reqs)",
                    y_label="decode t/s",
                    title="throughput vs concurrency"))

            # TTFT mean + p95 by concurrency.
            ttft_mean_pts = [(N, tput[N]["ttft_ms_mean"])
                             for N in concurrencies
                             if tput.get(N, {}).get("ttft_ms_mean") is not None]
            ttft_p95_pts = [(N, tput[N]["ttft_ms_p95"])
                            for N in concurrencies
                            if tput.get(N, {}).get("ttft_ms_p95") is not None]
            ttft_series = []
            if ttft_mean_pts:
                ttft_series.append({"label": "mean",
                                    "x": [p[0] for p in ttft_mean_pts],
                                    "y": [p[1] for p in ttft_mean_pts]})
            if ttft_p95_pts:
                ttft_series.append({"label": "p95",
                                    "x": [p[0] for p in ttft_p95_pts],
                                    "y": [p[1] for p in ttft_p95_pts]})
            if ttft_series:
                parts.append(bench_data.line_chart(
                    ttft_series, width=w, height=h,
                    x_label="concurrency (parallel reqs)",
                    y_label="TTFT ms",
                    title="TTFT vs concurrency"))

        # When concurrency varies, the per-preset facets below would mix all
        # concurrency levels — uninformative. Filter to the lowest concurrency
        # (canonical single-user view) for the preset charts, and tell the
        # viewer that's what they're seeing.
        canonical_n = concurrencies[0] if len(concurrencies) > 1 else None
        if canonical_n is not None:
            preset_rows = [r for r in rows if r.get("concurrency", 1) == canonical_n]
            parts.append(f"[dim](per-preset charts below filtered to c={canonical_n} "
                         f"— see concurrency overlay above for the multi-user story)[/]")
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


class ABPanel(Static):
    """Overlay charts + system-diff table across pinned runs."""

    def update_pinned(self, pinned: list[bench_data.RunIndex], width: int, height: int) -> None:
        if len(pinned) < 2:
            self.update("[dim]pin two or more runs (space) and press [b]c[/b] to compare[/dim]")
            return

        loads = [cmp_mod.load(p.path) for p in pinned]
        w = max(50, width - 6)
        h = max(12, (height - 16))

        # overlay decode_vs_gen — one series per pinned run (means across context sizes)
        from statistics import mean as _mean
        overlay_series = []
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
                overlay_series.append({"label": l.label, "x": xs, "y": ys})

        body = bench_data.line_chart(
            overlay_series, width=w, height=h,
            x_label="gen tokens", y_label="decode t/s",
            title="decode overlay (mean across context sizes)")

        # system diff
        if len(loads) == 2:
            diff = bench_data.system_diff(loads[0].system, loads[1].system)
            body += "\n\n[b]system diff[/b]\n"
            if not diff:
                body += "  (none captured)\n"
            else:
                body += f"  {'field':<14}  {'A':<30}  B\n"
                for f, va, vb in diff:
                    va_s = va if len(va) < 28 else va[:25] + "..."
                    vb_s = vb if len(vb) < 28 else vb[:25] + "..."
                    body += f"  {f:<14}  {va_s:<30}  {vb_s}\n"

            speedup = cmp_mod.compute_speedup(loads)
            safe = cmp_mod.configs_equivalent(loads)
            if speedup is not None:
                tag = "[b yellow]speedup[/b yellow]" if safe else "[b red]ratio (apples-to-oranges)[/b red]"
                body += f"\n  {tag}: [b]{speedup:.2f}×[/b]\n"

        self.update(body)


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
        Binding("space", "toggle_pin", "pin"),
        Binding("c", "show_ab", "compare"),
        Binding("p", "open_png", "PNG"),
        Binding("slash", "focus_filter", "/filter"),
        Binding("f1", "toggle_presenter", "presenter"),
        Binding("1", "tab('summary')", "summary"),
        Binding("2", "tab('charts')", "charts"),
        Binding("3", "tab('raw')", "raw"),
        Binding("4", "tab('ab')", "A/B"),
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
                    with TabPane("A/B", id="ab"):
                        with VerticalScroll():
                            yield ABPanel(id="ab_panel")
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
                    ts_local = bench_data.format_timestamp_local(r.timestamp, with_tz=False)
                    label = (f"{pin}{ts_local}  {r.rig_label}  "
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

    def _refresh_detail(self) -> None:
        size = self.size
        w, h = size.width - 42, size.height - 6  # rough right-pane size
        self.query_one("#summary_panel", SummaryPanel).update_run(self.selected, self.presenter)
        self.query_one("#charts_panel", ChartsPanel).update_run(self.selected, w, h)
        self.query_one("#raw_panel", RawPanel).update_run(self.selected)
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
            self.notify("move cursor onto a run leaf first", severity="warning")
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

# ======================================================================
# sysinfo.py
# ======================================================================

"""System manifest capture — CPU, RAM, OS, GPUs, llama.cpp build info.

All helpers are best-effort: missing tools produce None / empty list, never raise.
Every external call is shelled out so a viewer reading the source can see exactly
where each number came from.
"""
from __future__ import annotations

import datetime
import json
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], timeout: float = 5.0, merge_stderr: bool = False) -> str | None:
    if not cmd or not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        if out.returncode != 0:
            return out.stdout or None
        return out.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def collect_host() -> dict:
    uname = platform.uname()
    distro = None
    os_release = _read("/etc/os-release") or ""
    m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', os_release, re.MULTILINE)
    if m:
        distro = m.group(1)
    return {
        "hostname": socket.gethostname(),
        "os": uname.system,
        "kernel": uname.release,
        "machine": uname.machine,
        "distro": distro,
        "python": platform.python_version(),
    }


def collect_cpu() -> dict:
    cpuinfo = _read("/proc/cpuinfo") or ""
    model = None
    m = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    if m:
        model = m.group(1).strip()
    cores_physical = None
    cores_logical = None
    max_mhz = None
    lscpu = _run(["lscpu"]) or ""
    for line in lscpu.splitlines():
        if line.startswith("Core(s) per socket:"):
            try:
                per = int(line.split(":")[1].strip())
                sockets_line = re.search(r"^Socket\(s\):\s*(\d+)", lscpu, re.MULTILINE)
                sockets = int(sockets_line.group(1)) if sockets_line else 1
                cores_physical = per * sockets
            except (ValueError, AttributeError):
                pass
        elif line.startswith("CPU(s):") and cores_logical is None:
            try:
                cores_logical = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("CPU max MHz:"):
            try:
                max_mhz = float(line.split(":")[1].strip())
            except ValueError:
                pass

    governor = None
    gov = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov:
        governor = gov.strip()

    return {
        "model": model,
        "cores_physical": cores_physical,
        "cores_logical": cores_logical,
        "max_mhz": max_mhz,
        "governor": governor,
    }


def collect_memory() -> dict:
    meminfo = _read("/proc/meminfo") or ""

    def _kb(key: str) -> int | None:
        m = re.search(rf"^{key}:\s*(\d+)\s*kB", meminfo, re.MULTILINE)
        return int(m.group(1)) if m else None

    total = _kb("MemTotal")
    avail = _kb("MemAvailable")
    return {
        "total_gb": round(total / 1024 / 1024, 2) if total else None,
        "available_gb": round(avail / 1024 / 1024, 2) if avail else None,
    }


_NVIDIA_FIELDS = [
    "index", "uuid", "name",
    "memory.total", "memory.free",
    "driver_version",
    "pstate", "power.limit",
    "clocks.sm", "clocks.mem",
    "pcie.link.gen.current", "pcie.link.width.current",
]


def collect_nvidia_gpus() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    query = ",".join(_NVIDIA_FIELDS)
    raw = _run([
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ])
    if not raw:
        return []

    # CUDA version from nvidia-smi banner
    cuda = None
    banner = _run(["nvidia-smi"]) or ""
    m = re.search(r"CUDA Version:\s*([\d.]+)", banner)
    if m:
        cuda = m.group(1)

    gpus = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(_NVIDIA_FIELDS):
            continue
        d = dict(zip(_NVIDIA_FIELDS, parts))

        def _i(k: str) -> int | None:
            v = d.get(k, "").strip()
            if not v or v == "[N/A]":
                return None
            try:
                return int(float(v))
            except ValueError:
                return None

        gpus.append({
            "index": _i("index"),
            "uuid": d.get("uuid"),
            "name": d.get("name"),
            "vram_total_mib": _i("memory.total"),
            "vram_free_mib": _i("memory.free"),
            "driver": d.get("driver_version"),
            "cuda": cuda,
            "pstate": d.get("pstate"),
            "power_limit_w": _i("power.limit"),
            "sm_clock_mhz": _i("clocks.sm"),
            "mem_clock_mhz": _i("clocks.mem"),
            "pcie_gen": _i("pcie.link.gen.current"),
            "pcie_width": _i("pcie.link.width.current"),
        })
    return gpus


def collect_rocm_gpus() -> list[dict]:
    if not shutil.which("rocm-smi"):
        return []
    raw = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram",
                "--showdriverversion", "--json"])
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    gpus = []
    for card_id, info in data.items():
        if not card_id.lower().startswith("card"):
            continue
        gpus.append({
            "index": card_id,
            "name": info.get("Card series") or info.get("Card model"),
            "vram_total_mib": info.get("VRAM Total Memory (B)"),
            "driver": info.get("Driver version"),
        })
    return gpus


def collect_llama_cpp(binary: str | None) -> dict:
    out: dict = {"binary": binary, "version_line": None, "git_commit": None, "build_flags": None}
    if not binary:
        return out
    raw = _run([binary, "--version"], timeout=10.0, merge_stderr=True)
    if not raw:
        return out
    raw = raw.strip()
    out["version_line"] = raw.splitlines()[0] if raw else None
    # llama-server --version prints something like:
    #   version: 4321 (abc1234)
    #   built with cc (GCC) ... for x86_64-linux-gnu
    m = re.search(r"\(([0-9a-f]{6,40})\)", raw)
    if m:
        out["git_commit"] = m.group(1)
    # Heuristic flag extraction — look for known build markers
    flags = []
    for marker in ("CUDA", "VULKAN", "ROCM", "METAL", "BLAS",
                   "FA_ALL_QUANTS", "RPC", "OPENMP"):
        if re.search(rf"\b{marker}\b", raw):
            flags.append(marker)
    out["build_flags"] = " ".join(flags) if flags else None
    return out


def capture(*, rig_label: str, selection: dict, config_snapshot: dict,
            binary: str | None) -> dict:
    """Top-level: build the full system manifest dict."""
    return {
        "rig_label": rig_label,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": collect_host(),
        "cpu": collect_cpu(),
        "memory": collect_memory(),
        "selection": selection,
        "gpus": collect_nvidia_gpus(),
        "rocm_gpus": collect_rocm_gpus(),
        "llama_cpp": collect_llama_cpp(binary),
        "config": config_snapshot,
    }


def write(manifest: dict, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "system.json"
    out.write_text(json.dumps(manifest, indent=2, default=str))
    return out


if __name__ == "__main__":
    # `python sysinfo.py [binary]` — print the manifest for a quick sanity check.
    binary = sys.argv[1] if len(sys.argv) > 1 else None
    m = capture(
        rig_label="adhoc",
        selection={"cuda_visible_devices": None, "tensor_split": None, "main_gpu": None},
        config_snapshot={},
        binary=binary,
    )
    print(json.dumps(m, indent=2, default=str))

# ======================================================================
# plot.py
# ======================================================================

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


def plot_prefill_vs_context(rows, system, out: Path) -> None:
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
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _footer(fig, system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out)
    plt.close(fig)


def plot_throughput_vs_concurrency(rows, system, out: Path) -> None:
    """Aggregate decode throughput across all concurrent requests in a batch,
    plotted against concurrency level. Shows where adding more clients stops
    helping (the throughput-curve knee)."""
    from collections import defaultdict
    # Group by batch_id → sum predicted_n, max wall_s
    by_batch: dict[str, dict] = defaultdict(lambda: {"toks": 0, "wall": 0.0, "N": 0})
    for r in rows:
        bid = r.get("batch_id")
        if not bid or r.get("predicted_n") is None or r.get("batch_wall_s") is None:
            continue
        b = by_batch[bid]
        b["toks"] += r["predicted_n"]
        b["wall"] = max(b["wall"], r.get("batch_wall_s") or 0)
        b["N"] = r.get("concurrency") or b["N"]
    # Group again by concurrency → list of aggregate t/s per batch
    by_n: dict[int, list[float]] = defaultdict(list)
    for b in by_batch.values():
        if b["wall"] > 0 and b["N"]:
            by_n[b["N"]].append(b["toks"] / b["wall"])
    if len(by_n) < 2:
        return  # nothing to compare
    levels = sorted(by_n)
    means = [sum(by_n[n]) / len(by_n[n]) for n in levels]
    mins = [min(by_n[n]) for n in levels]
    maxs = [max(by_n[n]) for n in levels]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(levels, means, marker="o", linewidth=2.5, label="aggregate decode t/s",
            color="#a5d6ff")
    ax.fill_between(levels, mins, maxs, alpha=0.2, color="#a5d6ff")
    # Also plot per-request mean t/s (decode_tps from each row, averaged at each N)
    per_req: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("decode_tps") and r.get("concurrency"):
            per_req[r["concurrency"]].append(r["decode_tps"])
    if len(per_req) >= 2:
        per_req_means = [sum(per_req[n]) / len(per_req[n]) for n in levels if n in per_req]
        ax.plot(levels, per_req_means, marker="s", linewidth=2,
                label="per-request decode t/s", linestyle="--", color="#ffd166")
    ax.set_xticks(levels)
    ax.set_xlabel("concurrency (parallel requests)")
    ax.set_ylabel("decode throughput (tokens/s)")
    ax.set_title("Throughput vs concurrency")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _footer(fig, system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out)
    plt.close(fig)


def plot_ttft_vs_concurrency(rows, system, out: Path) -> None:
    """TTFT distribution by concurrency level — shows whether parallel requests
    queue (TTFT spikes when N > server's --parallel) and the tail story."""
    from collections import defaultdict
    by_n: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("ttft_ms") is not None and r.get("concurrency"):
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
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _footer(fig, system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out)
    plt.close(fig)


def plot_ttft_vs_context(rows, system, out: Path) -> None:
    """TTFT (time-to-first-token) vs context size — dominated by prefill cost,
    so should scale ~linearly with ctx. Mostly insensitive to gen_size since
    we measure to the first chunk."""
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
    fig.savefig(out)
    plt.close(fig)


def plot_decode_vs_gen(rows, system, out: Path) -> None:
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
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _footer(fig, system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out)
    plt.close(fig)


def plot_heatmap_decode(rows, system, out: Path) -> None:
    cells = bench_data.aggregate_by_cell(rows)
    if not cells:
        return
    context_sizes = sorted({ps for (ps, _) in cells.keys()})
    gen_sizes = sorted({gs for (_, gs) in cells.keys()})
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
    fig.savefig(out)
    plt.close(fig)


def plot_draft_accept(rows, system, out: Path) -> bool:
    draft_rows = [r for r in rows if r.get("draft_n")]
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
    fig.savefig(out)
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
    fig.savefig(out)
    plt.close(fig)
    return True


def write_summary_md(rows, system, summary: dict, out: Path) -> None:
    cells = bench_data.aggregate_by_cell(rows)
    # Fall back to computing headline numbers from rows if summary.json was empty.
    if not summary:
        decode_vals = [r["decode_tps"] for r in rows if r.get("decode_tps")]
        prefill_vals = [r["prefill_tps"] for r in rows if r.get("prefill_tps")]
        accept_vals = [r["draft_accepted"] / r["draft_n"] for r in rows
                       if r.get("draft_n") and r.get("draft_accepted") is not None]
        summary = {
            "prefill_tps_mean": mean(prefill_vals) if prefill_vals else None,
            "decode_tps_mean": mean(decode_vals) if decode_vals else None,
            "accept_rate_mean": mean(accept_vals) if accept_vals else None,
        }
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
    if summary.get("prefill_tps_mean"):
        lines.append(f"- prefill mean: **{summary['prefill_tps_mean']:.1f} t/s**")
    if summary.get("decode_tps_mean"):
        lines.append(f"- decode mean: **{summary['decode_tps_mean']:.1f} t/s**")
    if summary.get("accept_rate_mean") is not None:
        lines.append(f"- draft accept mean: **{summary['accept_rate_mean']*100:.1f}%**")
    lines.append("")
    lines.append("## Per-cell means")
    lines.append("")
    lines.append("| ctx | gen | prefill t/s | decode t/s | accept % | n |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for (cs, gs) in sorted(cells.keys()):
        c = cells[(cs, gs)]
        prefill = (f"{c['prefill_mean']:.1f} ±{c['prefill_std']:.1f}"
                   if c["prefill_mean"] is not None else "-")
        decode = (f"{c['decode_mean']:.1f} ±{c['decode_std']:.1f}"
                  if c["decode_mean"] is not None else "-")
        accept = (f"{c['accept_mean']*100:.1f}"
                  if c["accept_mean"] is not None else "-")
        lines.append(f"| {cs} | {gs} | {prefill} | {decode} | {accept} | {c['n']} |")
    out.write_text("\n".join(lines))


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
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", title="preset")
    _footer(fig, system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out)
    plt.close(fig)


def write_summary_md_multi(rows, system, summary: dict, out: Path) -> None:
    """Extended summary.md with per-preset AND per-concurrency facets."""
    presets = sorted({r.get("preset") for r in rows if r.get("preset")})
    concurrencies = sorted({r.get("concurrency", 1) for r in rows})

    # Always write the headline summary first.
    write_summary_md(rows, system, summary, out)

    # Nothing more to add for trivial single-preset, single-concurrency runs.
    if len(presets) <= 1 and len(concurrencies) <= 1:
        return

    lines = [out.read_text().rstrip(), ""]

    # ---- Throughput by concurrency (the headline workgroup view) ----
    if len(concurrencies) > 1:
        tput = bench_data.aggregate_throughput_by_concurrency(rows)
        lines.append("## Throughput by concurrency")
        lines.append("")
        lines.append("Aggregate decode throughput is `sum(tokens across all parallel "
                     "requests in a batch) / batch_wall_s`. Per-request rate is the "
                     "token-weighted average of each individual request's decode_tps.")
        lines.append("")
        lines.append("| c | aggregate t/s | per-req t/s | prefill t/s | TTFT mean ms | TTFT p95 ms | batches | reqs |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for N in concurrencies:
            m = tput.get(N, {})
            agg = f"{m['aggregate_tps_mean']:.1f}" if m.get("aggregate_tps_mean") is not None else "-"
            per = f"{m['per_request_tps_mean']:.1f}" if m.get("per_request_tps_mean") is not None else "-"
            pre = f"{m['prefill_tps_mean']:.1f}" if m.get("prefill_tps_mean") is not None else "-"
            ttm = f"{m['ttft_ms_mean']:.0f}" if m.get("ttft_ms_mean") is not None else "-"
            ttp = f"{m['ttft_ms_p95']:.0f}" if m.get("ttft_ms_p95") is not None else "-"
            lines.append(f"| {N} | **{agg}** | {per} | {pre} | {ttm} | {ttp} | "
                         f"{m.get('n_batches', 0)} | {m.get('n_requests', 0)} |")
        lines.append("")

    # ---- Per-preset breakdown ----
    if len(presets) > 1:
        lines.append("## Per-preset breakdown")
        lines.append("")
        from statistics import mean as _mean
        for preset in presets:
            sub = [r for r in rows if r.get("preset") == preset]
            if not sub:
                continue
            cells = bench_data.aggregate_by_cell(sub)
            lines.append(f"### `{preset}`  ({len(sub)} rows)")
            lines.append("")
            _emit_cell_table(lines, cells)
            lines.append("")

    # ---- Per-concurrency × per-preset detail ----
    if len(concurrencies) > 1:
        lines.append("## Per-concurrency cell detail")
        lines.append("")
        for N in concurrencies:
            sub = [r for r in rows if r.get("concurrency", 1) == N]
            if not sub:
                continue
            lines.append(f"### c = {N}  ({len(sub)} rows)")
            lines.append("")
            if len(presets) > 1:
                for preset in presets:
                    sub2 = [r for r in sub if r.get("preset") == preset]
                    if not sub2:
                        continue
                    cells = bench_data.aggregate_by_cell(sub2)
                    lines.append(f"**preset = `{preset}`** ({len(sub2)} rows)")
                    lines.append("")
                    _emit_cell_table(lines, cells)
                    lines.append("")
            else:
                cells = bench_data.aggregate_by_cell(sub)
                _emit_cell_table(lines, cells)
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

# ======================================================================
# compare.py
# ======================================================================

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
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _footer(fig, loads[0].system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out)
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
    ax.grid(True, alpha=0.3)
    ax.legend()
    _footer(fig, loads[0].system)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out)
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

