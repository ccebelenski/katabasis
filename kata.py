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
    """Return (argv, env_overrides, notes). Applies hardware.* to args/env,
    and auto-bumps --parallel, --ctx-size, and --kv_unified so the server
    has enough slot / KV budget for the ramp's max_concurrency cap."""
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

    # Auto-bump --parallel to max_concurrency (the ramp's safety cap).
    # Never reduce a user-set value.
    required_parallel = bench_data.get_max_concurrency(sweep)
    if required_parallel > 1:
        try:
            existing_parallel = int(flat.get("--parallel", flat.get("-np", 1)))
        except (TypeError, ValueError):
            existing_parallel = 1
        if existing_parallel < required_parallel:
            args = [e for e in args
                    if not (isinstance(e, dict) and set(e.keys()) & {"--parallel", "-np"})
                    and e not in ("--parallel", "-np")]
            args.append({"--parallel": required_parallel})
            notes["parallel_override"] = (
                f"--parallel auto-bumped from {existing_parallel} to "
                f"{required_parallel} to match max_concurrency"
            )
        flat = bench_data.server_args_to_dict(args)

    # Auto-size --ctx-size. With --kv_unified the slots share one KV budget,
    # so total demand = max_c × (max_ctx + max_gen). 10% headroom covers
    # BOS/EOS, MTP draft tokens, and alignment padding.
    context_sizes = bench_data.get_context_sizes(sweep)
    gen_sizes = sweep.get("gen_sizes") or []
    if context_sizes and gen_sizes:
        max_ctx = max(context_sizes)
        max_gen = max(gen_sizes)
        max_c = required_parallel
        required_ctx_size = int((max_ctx + max_gen) * max_c * 1.10)
        try:
            existing_ctx_size = int(flat.get("--ctx-size", 0) or 0)
        except (TypeError, ValueError):
            existing_ctx_size = 0
        if existing_ctx_size < required_ctx_size:
            args = [e for e in args
                    if not (isinstance(e, dict) and "--ctx-size" in e)
                    and e != "--ctx-size"]
            args.append({"--ctx-size": required_ctx_size})
            notes["ctx_size_override"] = (
                f"--ctx-size auto-sized to {required_ctx_size} "
                f"(= ({max_ctx} + {max_gen}) × {max_c} × 1.10) — was "
                f"{existing_ctx_size or 'unset'}"
            )
            flat = bench_data.server_args_to_dict(args)

    # Inject --kv_unified when running multi-slot, if not already set. Unified
    # KV lets slots share the budget dynamically — required for the ramp's
    # budget math above to be honored by the server.
    if required_parallel > 1 and "--kv_unified" not in flat and "--kv-unified" not in flat:
        args.append("--kv_unified")
        notes["kv_unified_added"] = (
            "--kv_unified injected (shared KV budget across slots — "
            "required for max_concurrency > 1)"
        )

    # Bump server log verbosity so the startup log emits buffer-size /
    # offload / KV-allocation lines the VRAM check parses. Default
    # verbosity=3 is too terse for diagnostic use. 5 brings back the
    # llm_load_tensors detail without flooding the log with per-token
    # spam (-v / log-all is overkill). User can override by setting
    # -lv / --verbosity / --log-verbosity in server.args explicitly.
    if not any(k in flat for k in ("-lv", "--verbosity", "--log-verbosity")):
        args.append({"-lv": 5})
        notes["verbosity_bumped"] = (
            "-lv 5 injected (richer startup log for VRAM diagnostics — "
            "set --verbosity in server.args to override)"
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


# ---------- rolling-ramp constants ------------------------------------------
#
# These are pinned to keep runs cross-comparable. Tuning them per-config would
# distort the stability bar / knee definition. Documented in
# project_design_decisions memory; do not expose to YAML.

# Min consecutive completed samples to evaluate stability at a given c level.
STABILITY_WINDOW_N = 20

# Advance to c+1 when the 95% CI half-width on the rolling window's mean
# aggregate decode_tps is below this fraction of the mean. Tight enough that
# the headline number is genuinely stable, loose enough to finish.
STABILITY_CI_REL_THRESHOLD = 0.10

# Discard the first N arrivals at each c level — they're transient (prior
# level's queue is draining, slot allocation churning). Counted in fired
# order, not completion order.
TRANSIENT_DISCARD_N = 5

# Consecutive-error abort threshold. If this many in-flight requests
# return exceptions in a row at one c level, the level exits early with
# mode="errors" rather than burning the rest of MAX_SAMPLES_PER_LEVEL.
# Audit ROBUSTNESS #5: protects against the silent-waste case when the
# server can't keep up at high c and starts dropping requests.
ERROR_ABORT_THRESHOLD = 5

# Hard cap per level if CI never converges. Cell wall time bound.
# Reduced from 60 → 30 (2026-06-06) after observing levels that ran to
# n=55+ on the Ada run when verify-mode failed AND CI never converged —
# typically heavy-ctx cells where TTFT variance pushes decode noise above
# the 10% CI threshold permanently. If 30 samples haven't converged, 60
# won't either; just stop wasting wall time on inherently noisy levels.
# The noisier mean from a smaller sample is acceptable because verify-
# mode-failure already signals "this level is hard to characterize."
MAX_SAMPLES_PER_LEVEL = 30

# Lower bound on per-req throughput as a fraction of the c=1 baseline. Acts
# as a backstop termination — for systems with no c_sat>1 bounce (pure c1/c
# degradation) the aggregate-decline check below never trips, so this floor
# is the final word. For c_sat>1 systems, aggregate-decline usually fires
# first and this floor is rarely reached.
USELESS_DECODE_FRACTION = 0.10

# Aggregate-decline termination. After the aggregate t/s curve has exceeded
# the c=1 baseline (which means we've actually found a c_sat>1 bounce — for
# pure c1/c regimes the curve never beats c=1 and this check never fires),
# count consecutive levels where aggregate drops below the running peak.
# Terminate when N consecutive declines occur — we're definitively past the
# useful concurrency point and further levels just confirm the descent.
#
# Suppresses two failure modes of the per-req-only floor:
#   (a) premature termination in the c=2..c_sat dip region (per-req is below
#       the floor briefly there because of the c1/c overhead, but the
#       aggregate hasn't peaked yet so we shouldn't stop)
#   (b) plateau false-negatives where per-req hovers near the floor without
#       crashing through cleanly (observed 2026-06-05 on Qwen3.6/GB10:
#       per-req plateaued at 2.6-3.2 t/s near a 2.89 floor, ramp climbed to
#       c=15 before hitting max_concurrency)
#
# The `c >= 4` guard ensures we have enough data points (c=1,2,3 + the
# current level) before considering termination — avoids over-reacting to
# early noise.
AGGREGATE_DECLINE_N = 2
AGGREGATE_DECLINE_MIN_C = 4

# Plateau-detection termination. Aggregate-decline (above) catches the case
# where the system clearly slides past peak; plateau-detection catches the
# subtler case where aggregate stays *within noise of peak* for several
# consecutive levels — we're not learning anything new, just burning samples.
# Triggered when N consecutive levels have aggregate within PLATEAU_TOL of
# the running peak. Same passed_c1_aggregate guard as decline (don't fire
# in the early dip region where aggregate hasn't yet beaten c=1 baseline).
AGGREGATE_PLATEAU_N = 3
AGGREGATE_PLATEAU_TOL = 0.05  # within 5% of peak counts as "still at peak"

# Saturation signal: in-flight is hard-capped at `concurrency` so the
# label is always honest (never measuring effective-c > requested c).
# We detect saturation by comparing the achieved fire rate to the target:
# when achieved < SATURATION_RATE_FRACTION × target over a recent window,
# the server can't sustain the target rate and the cap is throttling fires.
# That's an informational label, NOT a stop signal — the ramp keeps going.
SATURATION_RATE_FRACTION = 0.90

# Prediction-shortcut knobs. After MIN_FULL_LEVELS levels are measured fully,
# we fit a per_req(c) model and run subsequent levels in verify mode — just
# VERIFY_WINDOW_N samples past the transient, compared to the prediction.
# If the verify window matches prediction within tolerance, the level is
# labeled "verified" and we move on. Otherwise we fall back to full
# measurement (something interesting is happening). User-disableable via
# sweep.predict_shortcut: false in YAML — useful when running CPU offload
# or anything else likely to deviate from a clean compute-bound curve.
#
# Tolerance is ASYMMETRIC by direction:
# - VERIFY_TOLERANCE_UNDER: per-req is BELOW prediction. This is unexpected
#   bad news — something is degrading worse than the model predicts. Tight
#   tolerance triggers full measurement so we can investigate.
# - VERIFY_TOLERANCE_OVER: per-req is ABOVE prediction. This is unexpected
#   GOOD news — the system is outperforming the c1/c model. For c_sat>1
#   systems (parallel-decode bounce, observed on Qwen3.6/GB10 2026-06-05),
#   per-req beats c1/c by +9-15% in the boost region. We don't need to
#   investigate this further; we already know what it is. Wider tolerance
#   accepts the level quickly and moves on. Without this, c_sat>1 cells
#   waste massive samples (1897 actual vs 1440 estimated in the 2026-06-05
#   run) because every boost level falls back to MAX_SAMPLES_PER_LEVEL=60.
MIN_FULL_LEVELS = 3
VERIFY_WINDOW_N = 8
VERIFY_TOLERANCE_UNDER = 0.10
VERIFY_TOLERANCE_OVER = 0.30


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


_MOE_OFFLOAD_FLAGS = {
    "-cmoe", "--cpu-moe",
    "-ncmoe", "--n-cpu-moe",
    "-cmoed", "--cpu-moe-draft", "--spec-draft-cpu-moe",
    "-ncmoed", "--n-cpu-moe-draft", "--spec-draft-n-cpu-moe",
}


def _has_moe_offload(server_args: list) -> bool:
    """Detect whether the user has explicitly opted into CPU expert
    offload for MoE models. When set, Host memory allocations are
    intentional (trading speed for footprint to fit larger MoE
    models on smaller cards) — NOT the silent-spillover failure mode
    we warn about for dense models."""
    if not server_args:
        return False
    for entry in server_args:
        if isinstance(entry, dict):
            for k in entry.keys():
                if k in _MOE_OFFLOAD_FLAGS:
                    return True
        elif isinstance(entry, str) and entry in _MOE_OFFLOAD_FLAGS:
            return True
    return False


def check_vram_budget(log_path: Path, console: Console,
                      moe_offload_intentional: bool = False) -> None:
    """Parse llama-server's startup log for VRAM-related signals after the
    server has come up healthy. The current llama-server build emits
    a different (sparser) log format than older versions — there's no
    explicit "weights on CPU vs GPU" line. We extract what we can:

    - device_info lines showing per-GPU total and free MiB at startup
    - Memory estimates: mmproj (multimodal), MTP draft context, prompt cache
    - Failure flags: OOM, fit-failed, fallback-to-CPU patterns

    Future iteration could sample GPU telemetry pre/post model-load to
    compute total VRAM used as a delta. For now this surfaces the
    estimates llama-server publishes plus a loud warning on detected
    failures.

    OOM during startup is handled implicitly: wait_for_health times out
    and this function isn't reached.
    """
    import re
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return
    devices: list[tuple[str, float, float]] = []   # (name, total_mib, free_mib)
    estimates: list[tuple[str, float]] = []        # (component, mib)
    warnings: list[str] = []
    # Final-state per-device layer count (after any -fit on adjustments).
    # Initial `load_tensors: layer N assigned to device X` lines reflect
    # the FIRST init pass; we track the post-fit decision separately via
    # `set ngl_per_device_high[N].n_layer=M` and `n_layer=N` in fit logs.
    fit_layers_on_device: dict[int, int] = {}
    # Most recent host memory breakdown row (model, context, compute MiB).
    # If non-trivial model OR context shows up on Host, that's CPU offload —
    # the bug we missed on the 2026-06-05 Ada run where -fit on spilled
    # ~10 GB of weights + 4 GB of context to host memory silently.
    host_breakdown: tuple[float, float, float] | None = None  # (model, ctx, compute)
    fit_active = False
    fit_reduced_by: float = 0.0
    for line in text.splitlines():
        # device_info: - CUDA0 : NVIDIA GB10 (124545 MiB, 120219 MiB free)
        m = re.search(r"- (\w+)\s*:\s*([^(]+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)", line)
        if m:
            devices.append((m.group(1).strip(), float(m.group(3)), float(m.group(4))))
            continue
        # estimated worst-case memory usage of mmproj is 1161.02 MiB
        # estimated memory usage of MTP context is 2596.06 MiB
        m = re.search(r"estimated (?:worst-case )?memory usage of ([\w ]+?) is\s+([\d.]+)\s*MiB", line)
        if m:
            estimates.append((m.group(1).strip(), float(m.group(2))))
            continue
        # prompt cache is enabled, size limit: 8192 MiB
        m = re.search(r"prompt cache is enabled,\s+size limit:\s*(\d+)\s*MiB", line)
        if m:
            estimates.append(("prompt cache (limit)", float(m.group(1))))
            continue
        # -fit on signals: server is trying to reshape the config to fit.
        # `cannot meet free memory target of X MiB, need to reduce ... by Y MiB`
        if "common_init_result: fitting params to device memory" in line:
            fit_active = True
            continue
        m = re.search(r"need to reduce device memory by\s+(\d+)\s*MiB", line)
        if m:
            fit_reduced_by = max(fit_reduced_by, float(m.group(1)))
            continue
        # Post-fit final layer placement decision:
        # `set ngl_per_device_high[0].n_layer=52` (52 of N layers on GPU 0)
        m = re.search(r"set ngl_per_device_high\[(\d+)\]\.n_layer=(\d+)", line)
        if m:
            fit_layers_on_device[int(m.group(1))] = int(m.group(2))
            continue
        # Memory breakdown rows — the truth after each init pass. Format:
        # | - CUDA0 (...) | TOTAL = FREE + (SELF = MODEL + CONTEXT + COMPUTE) + UNACCOUNTED |
        # | - Host         |                  HOSTSUM = MODEL + CONTEXT + COMPUTE          |
        # The Host row is the smoking gun for CPU offload — if MODEL or
        # CONTEXT columns are non-trivial, weights or KV are off-GPU.
        m = re.search(
            r"-\s*Host\s*\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)",
            line,
        )
        if m:
            host_breakdown = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
            continue
        # Failure patterns.
        low = line.lower()
        if "out of memory" in low or "cuda error: out of memory" in low:
            warnings.append("OOM detected during model load")
        elif "falling back to cpu" in low or "fallback to cpu" in low:
            warnings.append("server fell back to CPU for some operation")
    # Detect CPU spillover from the Host breakdown row. The compute column
    # legitimately has a non-zero Host buffer (CUDA_Host pinned scratch);
    # what's NOT legitimate is model or context columns being non-trivial.
    # EXCEPT when MoE expert offload is intentional (-cmoe/--n-cpu-moe) —
    # then Host model allocation is the configured behavior, not a bug.
    if host_breakdown is not None:
        host_model, host_ctx, _host_compute = host_breakdown
        if host_model > 500 or host_ctx > 500:
            if moe_offload_intentional:
                # Surface the numbers as info, not as a problem.
                parts_extra = []
                if host_model > 500:
                    parts_extra.append(f"{host_model/1024:.1f} GiB model")
                if host_ctx > 500:
                    parts_extra.append(f"{host_ctx/1024:.1f} GiB context")
                console.print(
                    f"[dim]MoE CPU offload (intentional): "
                    f"{' + '.join(parts_extra)} on Host[/dim]"
                )
            else:
                warnings.append(
                    f"-fit on spilled to Host memory: {host_model/1024:.1f} GiB "
                    f"model + {host_ctx/1024:.1f} GiB context — decode will be "
                    f"5-10× slower than fully-on-GPU; benchmark NOT comparable "
                    f"to on-GPU runs. Re-run with --fit off and reduce "
                    f"max_concurrency or context_sizes until it fits. (If this "
                    f"is an MoE model with intentional expert offload, set "
                    f"-cmoe / --n-cpu-moe N in server.args.)"
                )
    if fit_active and fit_reduced_by > 0:
        # Even if Host ended up clean, the fact that fit had to reduce
        # memory means we're near the edge — note it for diagnostic value.
        warnings.append(
            f"-fit on activated and reshaped config to fit "
            f"(reduced by {fit_reduced_by/1024:.1f} GiB) — run with "
            f"--fit off to ensure deterministic config across rigs"
        )
    parts: list[str] = []
    if devices:
        gpu_devs = [(n, t, f) for n, t, f in devices if n != "CPU"]
        for name, total, free in gpu_devs:
            parts.append(f"{name} {free/1024:.1f}/{total/1024:.1f} GiB free at startup")
    if fit_layers_on_device:
        layer_str = ", ".join(f"{n} on GPU{d}" for d, n in sorted(fit_layers_on_device.items()))
        parts.append(f"post-fit layers: {layer_str}")
    for component, mib in estimates:
        parts.append(f"{component} ~{mib/1024:.2f} GiB est")
    if warnings:
        for w in warnings:
            console.print(f"[bold red]⚠ {w}[/bold red]")
    if parts:
        console.print(f"[dim]VRAM: {' · '.join(parts)}[/dim]")


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


def run_completion(endpoint: str, prompt: str, n_predict: int) -> dict:
    """POST to native /completion (streaming) and return the final
    timings/metadata. Streams in the request to keep the connection
    alive and to pick up the per-event timings block; we don't expose
    per-token / per-rate callbacks because the dashboard switched from
    a token-stream panel to an event log (#17). Token text is not
    retained — only the final timings are surfaced.

    Returns dict with: prefill_tps, decode_tps, ttft_ms, prompt_n,
                       predicted_n, draft_n, draft_accepted, raw_timings,
                       wall_s
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
    final: dict[str, Any] = {}
    t_start = time.time()
    t_first_token: float | None = None
    # Diagnostic capture: ring buffer of last raw lines + decode-error counter,
    # surfaced only when raw_timings ends up empty (see _debug_capture below).
    _dbg_tail: deque = deque(maxlen=8)
    _dbg_decode_errors = 0
    _dbg_stop_seen = False
    _dbg_total_lines = 0
    # `pending` holds the in-progress SSE event body across iter_lines yields.
    # llama-server occasionally emits literal `\n` bytes inside JSON string
    # values (e.g. the echoed prompt in generation_settings on long requests),
    # which splits a single event across multiple iter_lines yields. We
    # accumulate until an SSE blank-line terminator (or a new `data:`) arrives
    # and only then attempt json.loads.
    pending = ""

    def _handle(evt: dict) -> None:
        nonlocal final, t_first_token, _dbg_stop_seen
        if evt.get("content") and t_first_token is None:
            t_first_token = time.time()
        if evt.get("stop") or evt.get("stopped_eos") or evt.get("stopped_limit") \
                or evt.get("stopped_word"):
            _dbg_stop_seen = True
            final = evt

    def _flush(body: str) -> None:
        nonlocal _dbg_decode_errors
        if not body or body == "[DONE]":
            return
        try:
            evt = json.loads(body)
        except json.JSONDecodeError:
            _dbg_decode_errors += 1
            return
        _handle(evt)

    with requests.post(url, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines(decode_unicode=True):
            _dbg_tail.append((raw_line or "")[:500])
            _dbg_total_lines += 1
            if not raw_line:
                # SSE event terminator (blank line) — flush any in-progress event.
                if pending:
                    body = pending[5:].lstrip() if pending.startswith("data:") else pending
                    pending = ""
                    _flush(body)
                continue
            if raw_line.startswith("data:"):
                # New event starts. Flush previous one if it didn't get a
                # blank-line terminator (defensive — well-formed SSE always does).
                if pending:
                    body = pending[5:].lstrip() if pending.startswith("data:") else pending
                    _flush(body)
                pending = raw_line
            elif pending:
                # Continuation: server embedded a literal \n byte inside a
                # JSON string value (echoed prompt in generation_settings on
                # long requests). Reassemble with the newline escaped so
                # json.loads sees a valid `\n` escape inside the string,
                # not a raw 0x0A (which JSON forbids in string values).
                pending += "\\n" + raw_line
        # End of stream — flush any final in-progress event.
        if pending:
            body = pending[5:].lstrip() if pending.startswith("data:") else pending
            _flush(body)
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

    result = {
        "prompt_n": prompt_n,
        "predicted_n": predicted_n,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "ttft_ms": ttft_ms,
        "draft_n": draft_n,
        "draft_accepted": draft_accepted,
        "wall_s": wall,
        "raw_timings": timings,
    }
    # Attach diagnostic capture only when timings came back empty — this is
    # how we learned what the SSE parser was missing on ctx=16384 code rows.
    if not timings:
        result["_debug_capture"] = {
            "stop_event_seen": _dbg_stop_seen,
            "json_decode_errors": _dbg_decode_errors,
            "total_raw_lines": _dbg_total_lines,
            "final_event_keys": sorted(final.keys()) if final else [],
            "final_event_truncated": {
                k: (str(v)[:300] if not isinstance(v, (int, float, bool, type(None))) else v)
                for k, v in (final.items() if final else [])
            },
            "last_raw_lines": list(_dbg_tail),
        }
    return result


# ---------- GPU telemetry ---------------------------------------------------


class GpuTelemetry:
    # 300 samples (~5 min at 1 Hz) — enough to fill the widest sparkline
    # cells × 2 dot columns even at the SPARK_W cap of 150 cells.
    HISTORY_LEN = 300

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
                        "_ts": time.time(),
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


def _fmt_ms_compact(ms: float) -> str:
    """Compact millisecond formatter for tight display contexts.
    < 1000   → "850"  (3-4 chars)
    1k-10k   → "3.5k" (4 chars, 1 decimal)
    >= 10k   → "14k"  (3-4 chars, no decimal)"""
    if ms < 1000:
        return f"{ms:.0f}"
    if ms < 10000:
        return f"{ms / 1000:.1f}k"
    return f"{ms / 1000:.0f}k"


# ---------- live dashboard --------------------------------------------------


_spark = bench_data.spark
_braille_bars = bench_data.braille_bars


class Dashboard:
    """rich Live layout, nvtop-style: per-GPU sparklines + bench sparklines + tokens."""

    SPARK_HISTORY = 300  # max data points we keep (deque maxlen)
    SPARK_RESERVED = 55  # chars reserved per row for label + cur/extra/mean
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
        # Per-concurrency sparkline deques. Each c level has its own history
        # that accumulates across sweep-loop revisits — so when c=8 comes
        # around again, the existing c=8 sparkline keeps growing instead of
        # resetting. Bench panel reads the active c's deque only.
        from collections import defaultdict
        self.recent_decode: defaultdict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.SPARK_HISTORY))
        self.recent_prefill: defaultdict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.SPARK_HISTORY))
        self.recent_ttft: defaultdict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.SPARK_HISTORY))
        self.recent_aggregate: defaultdict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.SPARK_HISTORY))
        self.recent_accept: defaultdict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.SPARK_HISTORY))

        # Cumulative per-concurrency stats — accumulate across ALL batches
        # at each c level (never cleared). Drives the inline "prior" row.
        # decode/prefill: list of (tokens, time) pairs for token-weighted mean.
        # ttft/agg/accept: list of values for arithmetic mean.
        self.cum_decode: defaultdict[int, list] = defaultdict(list)
        self.cum_prefill: defaultdict[int, list] = defaultdict(list)
        self.cum_ttft: defaultdict[int, list] = defaultdict(list)
        self.cum_agg: defaultdict[int, list] = defaultdict(list)
        self.cum_accept: defaultdict[int, list] = defaultdict(list)

        # Currently-active concurrency level — drives which per-c deque the
        # bench panel renders. Updated on each row arrival.
        self.last_c: int | None = None
        # In-flight slot count over time. Pushed from _run_level on every
        # fire and harvest. Rendered as a contrasting-color overlay on
        # the bottom row of the decode sparkline so the "stab and gap"
        # pattern at high c gains a continuous "yes, work is happening"
        # signal. The healthy pattern: ramp up to c quickly when a level
        # starts, drop to 0 at each completion cluster. Stuck-at-c =
        # stalled. (User's "in-flight overlay" idea, 2026-06-06.)
        self.recent_inflight: deque = deque(maxlen=4000)
        # Most recently reported in-flight count — used to synthesize
        # CONTINUOUS samples at render time. Without this, recent_inflight
        # only has events at fire/harvest boundaries, leaving most
        # render buckets blank ("smooshed to the right" bug, 2026-06-06).
        self._last_in_flight: int = 0
        # token_buf removed (along with token-stream panel) — see #17.
        # Event log — replaces the (broken) token-stream panel with the
        # actual diagnostic narrative: cell starts, level completions,
        # knee labels, termination reasons, fit results, VRAM warnings.
        # Persistent on disk as events.log JSONL when events_log_path is set.
        # maxlen matches the panel's display capacity (last 10 lines shown).
        self.event_buf: deque[dict] = deque(maxlen=10)
        self.events_log_path: Path | None = None
        self._events_log_fh = None
        self._events_log_lock = threading.Lock()

        # Sparkline render window — seconds of wall-clock history shown.
        # All sparklines (bench AND gpu) share the same window so column N
        # in any sparkline represents the same wall-clock interval.
        self.SPARKLINE_WINDOW_S = 120.0

        # Dashboard creation time — used as t=0 baseline for very early
        # snapshots before SPARKLINE_WINDOW_S of history exists.
        self._t_dashboard_start = time.time()
        # Shared render window for all sparklines. Initialized for the very
        # first render before update() runs; subsequent updates refresh it.
        self._render_window: tuple[float, float] = (
            self._t_dashboard_start - self.SPARKLINE_WINDOW_S,
            self._t_dashboard_start,
        )

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
        # Two unhappy cases the audit (ROBUSTNESS #1, #2) flagged:
        #   - UUID-form CUDA_VISIBLE_DEVICES (e.g. "GPU-abc123...") would
        #     silently produce _selected=set() and render no GPUs. Now
        #     we treat any non-int token as "selection by UUID I can't
        #     map" and fall back to _selected=None (show all manifest
        #     GPUs) so the operator sees telemetry rather than nothing.
        #   - Integer indices not present in the manifest (e.g. cvd="2"
        #     on a single-GPU host) produce an empty selection. Warn
        #     loudly so the operator catches the mismatch.
        sel = (system.get("selection") or {}).get("cuda_visible_devices")
        self._selected: set[int] | None = None
        if sel is not None:
            tokens = [s.strip() for s in str(sel).split(",") if s.strip()]
            int_indices = {int(t) for t in tokens if t.isdigit()}
            if not tokens or any(not t.isdigit() for t in tokens):
                # UUID-form or mixed — show all manifest GPUs as a safe
                # default and let the GPU panel be informative.
                self._selected = None
            else:
                self._selected = int_indices
                manifest_indices = {int(g.get("index", -1))
                                    for g in (system.get("gpus") or [])}
                missing = int_indices - manifest_indices
                if missing:
                    self.console.print(
                        f"[bold red]⚠ cuda_visible_devices selects GPU(s) "
                        f"{sorted(missing)} not present in system manifest "
                        f"(have {sorted(manifest_indices)}) — telemetry + "
                        f"dashboard GPU panel will be empty.[/bold red]"
                    )

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

    def _spark_w(self) -> int:
        """Sparkline width in cells, sized to fit the current terminal.

        Recomputed on every render so the dashboard adapts to terminal
        resizes mid-run. Caps at 150 cells to keep memory + render time
        bounded on very wide displays; floor at 20 to remain useful on
        narrow ones.
        """
        # Console.size queries the *current* terminal dimensions.
        width = max(20, self.console.size.width - self.SPARK_RESERVED - 4)
        return max(20, min(150, width))

    def _bench_panel_height(self) -> int:
        # title border (2)
        # + by-c summary row (wrapped to N visual lines when many c levels)
        #   + 1 separator (only when any data seen)
        # + metric blocks × BENCH_CHART_H rows each
        prior_rows = self._prior_row_lines()
        if prior_rows:
            prior_rows += 1  # separator below
        active_c = self.last_c if self.last_c is not None else 1
        n_metrics = (
            2  # prefill + decode (always present)
            + (1 if self.recent_aggregate.get(active_c) else 0)
            + (1 if self.recent_ttft.get(active_c) else 0)
            + (1 if self.recent_accept.get(active_c) else 0)
        )
        return 2 + prior_rows + n_metrics * self.BENCH_CHART_H

    def _build_layout(self) -> Layout:
        root = Layout()
        root.split_column(
            Layout(name="header", size=3),
            Layout(name="rig", size=6),
            Layout(name="gpu", size=self._gpu_panel_height()),
            Layout(name="bench", size=self._bench_panel_height()),
            Layout(name="events", size=10),
        )
        return root

    # ---- panels ------------------------------------------------------------

    def _header(self) -> Panel:
        name = self.cfg.get("name", "benchmark")
        cur = self.current or {}
        cur_c = cur.get("concurrency", 1)
        target_rate = cur.get("target_rate_hz")
        # Compute current-c prediction. Prefer the hyperbolic fit
        # (built from token-weighted per-c samples in self.cum_decode)
        # so the live preview matches what the post-run aggregator
        # will say; fall back to naive c1/c if we don't have enough
        # data yet for the fit (need >=2 levels at c>=2). Audit
        # CORRECTNESS #3 caught the previous c_sat=1 fallback.
        pred_str = ""
        if cur_c > 1:
            measured = []
            for c_lvl, pairs in self.cum_decode.items():
                if not pairs:
                    continue
                toks = sum(t for t, _ in pairs)
                tsum = sum(d for _, d in pairs)
                if tsum > 0:
                    measured.append((c_lvl, toks / tsum))
            fit = bench_data.fit_ramp_per_req(measured) if measured else None
            pred = bench_data.predict_ramp_per_req(cur_c, fit) if fit else None
            if pred is None and measured:
                # Fall back to c1/c from the c=1 baseline only.
                c1_val = next((v for c_, v in measured if c_ == 1), None)
                if c1_val is not None:
                    pred = c1_val / cur_c
            if pred is not None:
                pred_str = f"  pred {pred:.1f} t/s"
        rate_str = f"  rate {target_rate:.2f} Hz" if target_rate else ""
        # Per-level sample count + termination mode hint. Tells the operator
        # whether the current level is converging fast (verify exit at ~13
        # samples) or grinding (stability check, up to MAX_SAMPLES_PER_LEVEL).
        # samples_done is set per-completion by _run_level via dashboard.update.
        samples_done = cur.get("samples_done")
        samples_expected = cur.get("samples_expected")
        samples_mode = cur.get("samples_mode", "")
        samples_str = ""
        if samples_done is not None:
            if samples_expected:
                samples_str = f"  n={samples_done}/{samples_expected}"
            else:
                samples_str = f"  n={samples_done}"
            if samples_mode:
                samples_str += f" ({samples_mode})"
        cur_str = ((f"{cur.get('preset','?')}  ctx={cur.get('context_size')} "
                    f"g={cur.get('gen_size')}  c={cur_c}"
                    f"{rate_str}{pred_str}{samples_str}")
                   if self.current else "idle")

        # Progress display. The estimate is naive — it can't model c_sat>1
        # cells which trip verify-mode fallback and inflate sample counts.
        # Once done exceeds the estimate, switch to bare-count display so the
        # bar doesn't visually lie at >100% (capped) for the back half of the
        # run. Honest is better than precise here.
        if self.total and self.done <= self.total:
            pct = self.done / self.total * 100.0
            bar_filled = int(pct / 100.0 * 30)
            bar = "█" * bar_filled + "░" * (30 - bar_filled)
            progress_chunk = (f"[{bar}] ", "bright_blue"), (f"{self.done}/{self.total}  ", "bright_white")
        else:
            # Full bar with a trailing tick to flag "past estimate, count is
            # now the source of truth."
            bar = "█" * 29 + "▶"
            est_str = f" (est {self.total})" if self.total else ""
            progress_chunk = (f"[{bar}] ", "bright_blue"), (f"{self.done}{est_str}  ", "bright_white")

        text = Text.assemble(
            (f"{name}  ", "bold cyan"),
            *progress_chunk,
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

        # Selected GPU(s) — pull the names + VRAM from the manifest for
        # whatever cuda_visible_devices resolves to. Without this, the
        # rig panel just shows "cvd=0" and the operator has to remember
        # which card is GPU 0 on a multi-GPU host.
        sel_set = self._selected
        gpu_rows: list[str] = []
        for g_meta in (m.get("gpus") or []):
            idx = g_meta.get("index")
            if sel_set is not None and (idx is None or int(idx) not in sel_set):
                continue
            name = _short_gpu_name(g_meta.get("name")) or "?"
            vram_gib = (g_meta.get("vram_total_mib") or 0) / 1024
            gpu_rows.append(f"GPU{idx}: [bright_white]{name}[/] "
                            f"({vram_gib:.0f}G)")
        gpu_str = "  ".join(gpu_rows) if gpu_rows else "[dim](no selected GPU)[/]"

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
            f"[dim]llama.cpp {(llc.get('git_commit') or '?')[:10]}[/dim]",
        )
        g.add_row("gpu", gpu_str)
        g.add_row("model", str(model)
                  + (f"  [dim]draft={draft}[/dim]" if draft else ""))
        g.add_row("flags", flags_str)
        return Panel(g, title="rig + model", border_style="magenta",
                     padding=(0, 1))

    def _metric_block(self, label: str, ts_values, lo: float | None, hi: float | None,
                      cur: str, peak: str, style: str, height: int) -> list[Text]:
        """Render one metric as height-row Braille chart with label/values on
        the first row, indent on remaining rows. ts_values is a list of
        (timestamp, value) tuples; the chart aligns to self._render_window."""
        t_start, t_end = self._render_window
        bars = bench_data.time_bucketed_braille_bars(
            ts_values, self._spark_w(), height,
            t_start, t_end, lo=lo, hi=hi,
        )
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

            def col(key: str) -> list[tuple[float, float]]:
                # Returns (timestamp, value) tuples for time-aligned rendering.
                return [(s["_ts"], s[key]) for s in hist
                        if s.get(key) is not None and "_ts" in s]

            util = col("util_gpu")
            pwr = col("power_w")
            vram = col("vram_used_mib")
            temp = col("temp_c")

            # Extract values-only for cur/peak labels.
            util_v = [v for _, v in util]
            pwr_v = [v for _, v in pwr]
            vram_v = [v for _, v in vram]
            temp_v = [v for _, v in temp]

            cur_u = f"{util_v[-1]:3.0f}%" if util_v else "  -%"
            pk_u = f"{max(util_v):3.0f}%" if util_v else "  -%"
            cur_p = f"{pwr_v[-1]:4.0f}W" if pwr_v else "   -W"
            pk_p = f"{max(pwr_v):4.0f}W" if pwr_v else "   -W"
            cur_v = f"{vram_v[-1]/1024:4.1f}G" if vram_v else "   -G"
            pk_v = f"{max(vram_v)/1024:4.1f}G" if vram_v else "   -G"
            cur_t = f"{temp_v[-1]:3.0f}°C" if temp_v else "  -°C"
            pk_t = f"{max(temp_v):3.0f}°C" if temp_v else "  -°C"

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
            rows += self._metric_block("temp", temp, 0, 90, cur_t, pk_t, "bright_red", h)
        if not rows:
            rows = [Text("(no GPU samples yet)", style="dim")]
        return Panel(Group(*rows), title="gpu", border_style="magenta",
                     padding=(0, 1))

    def _bench_panel(self) -> Panel:
        from statistics import mean

        t_start, t_end = self._render_window

        def block(label: str, ts_vals, unit: str, style: str,
                  lo: float | None, hi: float | None,
                  fmt: str = "{:7.1f}", extra: str | None = None,
                  overlay_ts_vals=None, overlay_style: str = "bright_magenta",
                  overlay_hi: float | None = None) -> list[Text]:
            # ts_vals: list of (timestamp, value) tuples.
            # overlay_ts_vals: optional secondary series rendered as a
            # single Braille bar row REPLACING the bottom row of the
            # main metric's chart, in `overlay_style` color. Used for
            # the in-flight slot count overlay — fills the visual gap
            # between completion-cluster spikes at high c without
            # adding a new vertical row (the metric's bottom row is
            # mostly empty at high c anyway since spikes saturate the
            # upper rows). The metric loses 1 row of vertical
            # resolution where overlay is shown.
            values = [v for _, v in ts_vals]
            cur = fmt.format(values[-1]) if values else "    -- "
            mn = fmt.format(mean(values)) if values else "    -- "
            # If overlay present, render main chart 1 row shorter so the
            # overlay claims the bottom row. Otherwise normal height.
            main_h = (self.BENCH_CHART_H - 1) if overlay_ts_vals else self.BENCH_CHART_H
            main_h = max(1, main_h)  # never go below 1
            bars = bench_data.time_bucketed_braille_bars(
                ts_vals, self._spark_w(), main_h,
                t_start, t_end, lo=lo, hi=hi,
            )
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
            if overlay_ts_vals:
                overlay_bars = bench_data.time_bucketed_braille_bars(
                    overlay_ts_vals, self._spark_w(), 1,
                    t_start, t_end, lo=0.0, hi=overlay_hi,
                )
                # Single row by construction (height=1).
                out.append(Text.assemble(("        ", "dim"),
                                          (overlay_bars[0], overlay_style)))
            return out

        # All current-cycle sparklines read from the active c's per-c deque.
        # Each entry is a (timestamp, value) tuple.
        active_c = self.last_c if self.last_c is not None else 1
        active_decode = list(self.recent_decode.get(active_c, []))
        active_prefill = list(self.recent_prefill.get(active_c, []))
        active_ttft = list(self.recent_ttft.get(active_c, []))
        active_aggregate = list(self.recent_aggregate.get(active_c, []))
        active_accept = list(self.recent_accept.get(active_c, []))

        # Inline ms/tok on the decode line — same data as decode t/s but in
        # latency form, which is often easier to read at a glance.
        latest_decode_val = active_decode[-1][1] if active_decode else 0
        ms_per_tok_cur = (
            f"{1000.0/latest_decode_val:5.2f} ms/t"
            if latest_decode_val > 0
            else None
        )

        rows: list[Text] = []

        # Pinned per-concurrency summaries — all c levels seen so far inline
        # on a SINGLE row (which wraps when many c levels accumulate, see
        # _prior_row_lines for the height-budgeting story). Cumulative across
        # sweep revisits, so revisiting c=8 just updates the same number
        # rather than appending a new entry.
        prior_row = self._prior_row_text()
        if prior_row is not None:
            rows.append(prior_row)
            rows.append(Text("─" * 60, style="dim"))

        rows += block("prefill", active_prefill, "t/s", "bright_green", 0, None)
        # In-flight overlay on decode: shows slot busyness continuously
        # in contrasting magenta, claiming the bottom row of the decode
        # chart (which is otherwise empty at high c where spikes dominate
        # the upper rows). Healthy pattern: bar ramps to active_c quickly
        # at level start, drops to 0 at each completion cluster. The
        # deque is populated continuously by Dashboard.update() at the
        # ticker rate (4 Hz), so buckets fill densely.
        active_inflight = list(self.recent_inflight)
        rows += block("decode", active_decode, "t/s", "bright_yellow", 0, None,
                      extra=ms_per_tok_cur,
                      overlay_ts_vals=active_inflight if active_inflight else None,
                      overlay_style="bright_magenta",
                      overlay_hi=float(active_c) if active_c else None)
        if active_aggregate:
            rows += block("agg t/s", active_aggregate, "t/s",
                          "bright_cyan", 0, None)
        if active_ttft:
            rows += block("TTFT", active_ttft, "ms", "bright_blue", 0, None,
                          fmt="{:7.0f}")
        if active_accept:
            rows += block("accept",
                          [(ts, v * 100) for ts, v in active_accept],
                          "%", "bright_magenta", 0, 100, "{:6.1f}")
        return Panel(Group(*rows), title="benchmark", border_style="yellow",
                     padding=(0, 1))

    # Event-level color mapping. Each maps to a Rich style applied to the
    # whole line. Keep colors distinct enough to scan rapidly.
    EVENT_COLORS = {
        "info": "cyan",
        "transition": "yellow",
        "peak": "bright_green",
        "success": "green",
        "warn": "red",
        "error": "bold red",
    }

    def _events_panel(self) -> Panel:
        # Panel is height=10 → ~8 rows of content. Show last 8 events
        # newest-at-bottom (chronological top-to-bottom, like a log tail).
        lines: list[Text] = []
        for ev in list(self.event_buf)[-8:]:
            ts = ev.get("ts_local", "")
            level = ev.get("level", "info")
            color = self.EVENT_COLORS.get(level, "cyan")
            msg = ev.get("msg", "")
            line = Text()
            line.append(f"{ts}  ", style="dim")
            line.append(msg, style=color)
            lines.append(line)
        body = Group(*lines) if lines else Text("(no events yet)", style="dim")
        return Panel(body, title="events", border_style="cyan",
                     padding=(0, 1), height=10)

    # ---- update + ticker ---------------------------------------------------

    def _prior_row_text(self) -> Text | None:
        """Build the by-c summary Text. Returns None when no priors yet.
        Same logic used by both `_bench_panel` (for rendering) and
        `_prior_row_lines` (for height calc) so the two never diverge.

        TTFT renders as `lo-hi` range when its min/max ratio exceeds
        TTFT_RANGE_RATIO_THRESHOLD (cells with very different ctx produce
        very different TTFTs; one collapsed median would lie). Other metrics
        render as a single value because they're ctx-stable across cells."""
        priors = self._prior_summaries()
        if not priors:
            return None
        active_c = self.last_c if self.last_c is not None else 1
        # Header encodes the TTFT format directly: ttft is shown as the
        # median, with (p25..p75) IQR in parens when there are >=2 samples.
        # The IQR is robust to single-event outliers but widens when noise
        # is pervasive — a tight IQR consistently means stable conditions;
        # a wide IQR means something is genuinely all over the place.
        header = ("[bold dim cyan]by c[/]    "
                  "[dim](dec/agg, ttft median(p25..p75))[/]")
        chunks = [header]
        for s in priors:
            marker = "[bold bright_white]*[/]" if s["c"] == active_c else " "
            dec = f"[yellow]{s['decode']:5.1f}[/]" if "decode" in s else "  -- "
            agg = f"[bright_cyan]{s['agg']:5.1f}[/]" if "agg" in s else "  -- "
            ttft = self._format_ttft_cell(s)
            chunks.append(f"  {marker}[bold]c{s['c']:<2}[/] {dec}/{agg}/{ttft}")
        return Text.from_markup("".join(chunks), overflow="fold")

    def _format_ttft_cell(self, s: dict) -> str:
        """Markup string for one c's TTFT value: `median(p25..p75)` when there
        are enough samples to compute quartiles, just the median otherwise.

        Format choice: IQR (interquartile range) rather than min/max because
        IQR is robust to single-event outliers but still widens for
        systemic noise or cross-cell variance. A one-time TTFT spike won't
        move the IQR; a run where TTFT is genuinely all-over-the-place will
        show it clearly."""
        if "ttft" not in s:
            return "      --"
        med_str = _fmt_ms_compact(s["ttft"])
        if "ttft_p25" in s and "ttft_p75" in s:
            return (f"[blue]{med_str}[/][dim]([/]"
                    f"[blue]{_fmt_ms_compact(s['ttft_p25'])}"
                    f"..{_fmt_ms_compact(s['ttft_p75'])}[/]"
                    f"[dim])[/]")
        return f"[blue]{med_str}[/]"

    def _prior_row_lines(self) -> int:
        """Number of visual lines the by-c row takes after wrap. 0 if empty.
        Without this, _bench_panel_height under-allocates and the wrap pushes
        the bottom benchmark chart off the panel — observed when many c
        levels accumulate (e.g. c=1..16 doesn't fit on one line at any
        practical terminal width)."""
        txt = self._prior_row_text()
        if txt is None:
            return 0
        # Panel content width: console width - panel borders (2) - h-padding (2).
        content_w = max(20, self.console.size.width - 4)
        plain_len = len(txt.plain)
        return max(1, (plain_len + content_w - 1) // content_w)

    def _prior_summaries(self) -> list[dict]:
        """One summary per concurrency level (including the currently-active
        one — its number updates live as new batches at that c finish).

        Aggregation choice per metric (see project_design_decisions memory):
        - decode/prefill: token-weighted ratio (sum_tokens/sum_time). Correct
          for rates regardless of cell mix; per-sample arithmetic mean would
          be inflated by short-EOS requests.
        - agg/accept: median across samples. Robust to outliers; honest
          central tendency when distributions cross cells.
        - ttft: median + min/max range. TTFT is entirely prefill so it's
          dominated by ctx, and cells with different ctx produce wildly
          different TTFTs. Showing range when cells span >2x exposes the
          cross-cell mix instead of hiding it behind a misleading central
          value. See feedback memory on "ctx-sensitivity drives aggregation".
        """
        out: list[dict] = []
        from statistics import median as _median
        # Union of c levels seen by any metric — usually all the same set.
        all_c = set(self.cum_decode) | set(self.cum_prefill) | set(self.cum_ttft) \
                | set(self.cum_agg) | set(self.cum_accept)
        for c in sorted(all_c):
            s: dict = {"c": c}
            dec_pairs = self.cum_decode.get(c, [])
            if dec_pairs:
                toks = sum(t for t, _ in dec_pairs)
                time = sum(d for _, d in dec_pairs)
                if time > 0:
                    s["decode"] = toks / time
            pre_pairs = self.cum_prefill.get(c, [])
            if pre_pairs:
                toks = sum(t for t, _ in pre_pairs)
                time = sum(d for _, d in pre_pairs)
                if time > 0:
                    s["prefill"] = toks / time
            # Aggregate is definitionally per_req × c. Deriving from the
            # token-weighted decode keeps it consistent with the decode
            # column (at c=1 they must be identical; at c>1 they must
            # scale by exactly c). A separate median of per-batch agg
            # values would diverge from decode at c=1 since the two
            # aggregators (token-weighted vs median) produce slightly
            # different numbers from the same samples.
            if "decode" in s:
                s["agg"] = s["decode"] * c
            if self.cum_ttft.get(c):
                ttft_vals = self.cum_ttft[c]
                s["ttft"] = _median(ttft_vals)
                # IQR (p25..p75) captures the middle 50% of samples. Robust
                # to single-event outliers — a one-time TTFT spike won't
                # widen the IQR, but pervasive noise or cross-cell variance
                # will. quantiles() needs at least 2 values; for 1 sample
                # there's no spread to show.
                if len(ttft_vals) >= 2:
                    from statistics import quantiles as _quantiles
                    q = _quantiles(ttft_vals, n=4)
                    s["ttft_p25"], s["ttft_p75"] = q[0], q[2]
            if self.cum_accept.get(c):
                s["accept"] = _median(self.cum_accept[c]) * 100.0
            if len(s) > 1:
                out.append(s)
        return out

    def attach_events_log(self, path: Path) -> None:
        """Wire a JSONL events.log file. Each log_event() call appends a
        structured row to this file in addition to the in-memory deque
        and console echo. Must be called before any events are logged.
        Safe to skip — events still work in-memory, just aren't durable."""
        self.events_log_path = path
        # Line-buffered so events flush per write; survives a SIGINT.
        self._events_log_fh = open(path, "w", buffering=1)

    def close_events_log(self) -> None:
        if self._events_log_fh is not None:
            try:
                self._events_log_fh.close()
            except Exception:
                pass
            self._events_log_fh = None

    def log_event(self, msg: str, level: str = "info",
                  type: str | None = None,
                  echo: bool = True,
                  **fields) -> None:
        """Record a diagnostic event. Goes three places:
        1. self.event_buf (deque) for the live dashboard panel.
        2. events.log JSONL (if attached) for durable post-run analysis.
        3. console.print (if echo=True, default) for terminal scrollback —
           preserves existing scroll behavior so nothing is lost relative
           to the old console.print-only world.

        `level` is one of EVENT_COLORS keys: info, transition, peak,
        success, warn, error. `type` is an optional event taxonomy slug
        (e.g. "cell_start", "level_complete", "termination") for
        programmatic filtering. `**fields` carry structured data that's
        opaque to the live display but available in events.log for
        weatherman / pattern queries.
        """
        # Local-time ISO for human readability + offset.
        now = time.time()
        ts_local = time.strftime("%H:%M:%S", time.localtime(now))
        ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now))
        record = {
            "ts": ts_iso,
            "ts_local": ts_local,
            "ts_unix": now,
            "level": level,
            "msg": msg,
        }
        if type is not None:
            record["type"] = type
        record.update(fields)
        self.event_buf.append(record)
        if self._events_log_fh is not None:
            with self._events_log_lock:
                try:
                    self._events_log_fh.write(json.dumps(record) + "\n")
                except Exception:
                    pass  # don't let logging break the run
        if echo:
            color = self.EVENT_COLORS.get(level, "cyan")
            self.console.print(f"[{color}]{msg}[/{color}]")

    def update_in_flight(self, count: int) -> None:
        """Record the current in-flight slot count for the in-flight
        overlay on the decode sparkline. Called from _run_level on
        every fire and every completion harvest. Also updates the
        _last_in_flight cache that the render method uses to synthesize
        continuous samples at render time (since fire/harvest events
        alone are sparse, leaving the chart "smooshed to the right")."""
        self.recent_inflight.append((time.time(), count))
        self._last_in_flight = count

    def update_progress(self, samples_done: int,
                        samples_expected: int | None = None,
                        samples_mode: str = "") -> None:
        """Push per-level progress to the header without disturbing other
        fields of self.current. Called from _run_level as each completion
        lands so the operator can see how close the level is to its exit
        condition (verify quota, stability window, or max_samples cap)."""
        if self.current is None:
            self.current = {}
        self.current["samples_done"] = samples_done
        if samples_expected is not None:
            self.current["samples_expected"] = samples_expected
        self.current["samples_mode"] = samples_mode

    def update(self, current: dict | None = None, row: dict | None = None) -> None:
        # Wall-clock timestamp for any sparkline samples we append below.
        # All sparklines share a common time axis; column N maps to the
        # same wall-clock interval across all metrics.
        now = time.time()
        # Continuous in-flight sample so the overlay bar fills buckets
        # even between fire/harvest events. Without this, the bar only
        # shows samples at those discrete events and looks "smooshed to
        # the right" because most render buckets are blank. The ticker
        # calls update() every 250ms → 4 Hz sampling rate → at a 120s
        # window, ~480 samples fill the bar densely.
        self.recent_inflight.append((now, self._last_in_flight))
        if current is not None:
            self.current = current
        if row is not None:
            self.done += 1
            row_c = row.get("concurrency", 1)
            self.last_c = row_c
            dec_tps = row.get("decode_tps")
            if dec_tps:
                self.recent_decode[row_c].append((now, dec_tps))
                if row.get("predicted_n"):
                    # (tokens, decode_time) for token-weighted cumulative mean.
                    self.cum_decode[row_c].append((row["predicted_n"],
                                                   row["predicted_n"] / dec_tps))
                # Aggregate = per-req × c. In the rolling ramp with the
                # in-flight cap, this is the honest "total work being done"
                # at level c. Closed-loop batch_id is gone, so we derive it
                # from each row instead of summing over a batch.
                agg = dec_tps * row_c
                self.recent_aggregate[row_c].append((now, agg))
                self.cum_agg[row_c].append(agg)
            pre_tps = row.get("prefill_tps")
            if pre_tps:
                self.recent_prefill[row_c].append((now, pre_tps))
                if row.get("prompt_n"):
                    self.cum_prefill[row_c].append((row["prompt_n"],
                                                    row["prompt_n"] / pre_tps))
            if row.get("ttft_ms") is not None:
                self.recent_ttft[row_c].append((now, row["ttft_ms"]))
                self.cum_ttft[row_c].append(row["ttft_ms"])
            if row.get("draft_n") and row.get("draft_accepted") is not None and row["draft_n"]:
                accept_rate = row["draft_accepted"] / row["draft_n"]
                self.recent_accept[row_c].append((now, accept_rate))
                self.cum_accept[row_c].append(accept_rate)
        # Shared render window — all sparklines (bench + GPU) align to this
        # wall-clock range, so column N in any sparkline = same time slice.
        self._render_window = (now - self.SPARKLINE_WINDOW_S, now)

        # Refresh dynamic panel sizes — bench and gpu both grow as new
        # metrics appear (agg t/s line after first batch, prior-c row,
        # second GPU coming online, etc.). Without this the Layout reserves
        # the initial-empty size from __init__ and clips later content.
        self.layout["gpu"].size = self._gpu_panel_height()
        self.layout["bench"].size = self._bench_panel_height()

        self.layout["header"].update(self._header())
        self.layout["rig"].update(self._rig_panel())
        self.layout["gpu"].update(self._gpu_panel())
        self.layout["bench"].update(self._bench_panel())
        self.layout["events"].update(self._events_panel())

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


# ---------- rolling-ramp driver --------------------------------------------


def _fit_per_req_model(levels: list[dict]) -> dict | None:
    """Fit the physically-motivated per_req(c) model from measured levels.
    Delegates to bench_data so the live verify-mode fit and the
    post-run analysis use the same math (no chance of drift)."""
    measured = [(lvl["c"], lvl["per_req_decode_tps"])
                for lvl in levels if lvl.get("per_req_decode_tps") is not None]
    return bench_data.fit_ramp_per_req(measured)


def _predict_per_req(c: int, fit: dict | None) -> float | None:
    """Predict per-req throughput at concurrency c under the fitted model."""
    return bench_data.predict_ramp_per_req(c, fit)


def _ci_relative_halfwidth(values: list[float]) -> float:
    """95% CI half-width as a fraction of the mean. Used by the ramp's
    stability test. Returns +inf if mean is zero or sample is too small."""
    if len(values) < 2:
        return float("inf")
    m = sum(values) / len(values)
    if m <= 0:
        return float("inf")
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    sd = var ** 0.5
    ci_half = 1.96 * sd / (len(values) ** 0.5)
    return ci_half / m


def _write_row(raw_f, raw_lock, result: dict, *, preset: str,
               context_size: int, gen_size: int, concurrency: int,
               target_rate_hz: float | None, arrival_wall_s: float,
               queue_depth_at_arrival: int, fired_idx: int,
               in_steady_state: bool) -> dict:
    """Write one rolling-ramp row and return the row dict (so the caller can
    feed the same dict to the dashboard). Shape diverges from the closed-loop
    schema — see project_design_decisions memory for the rename log."""
    row = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "preset": preset,
        "context_size": context_size,
        "gen_size": gen_size,
        "concurrency": concurrency,
        "fired_idx": fired_idx,
        "target_rate_hz": target_rate_hz,
        "arrival_wall_s": arrival_wall_s,
        "queue_depth_at_arrival": queue_depth_at_arrival,
        "in_steady_state": in_steady_state,
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
    if result.get("_debug_capture") is not None:
        row["_debug_capture"] = result["_debug_capture"]
    with raw_lock:
        raw_f.write(json.dumps(row) + "\n")
        raw_f.flush()
    return row


def _run_level(*, endpoint: str, prompt: str, gen_size: int,
               preset: str, context_size: int, concurrency: int,
               target_rate_hz: float | None, cell_t0: float,
               raw_f, raw_lock, dashboard, console: Console,
               verify_prediction: float | None = None) -> dict:
    """Run one c level. Open-loop arrival metronome at target_rate_hz
    (None for c=1 — fire single-stream as previous request finishes).

    In-flight is capped at `concurrency`: if the target metronome time
    arrives while c requests are still in flight, we defer the fire until
    a completion frees a slot. This keeps the label "c=N" honest (at most
    N in flight) without artificially hammering — pre-saturation it stays
    open-loop with rate=target; at saturation it naturally becomes "fill
    slots as they free" (sustained-N closed-loop semantics).

    If `verify_prediction` is set, the level exits early once VERIFY_WINDOW_N
    steady-state samples are in AND their mean per_req is within the
    asymmetric verify tolerance (VERIFY_TOLERANCE_UNDER for shortfalls below
    prediction, VERIFY_TOLERANCE_OVER for surpluses above — see those
    constants for why). Mode is set to "verified" in that case. If outside
    tolerance, falls through to full measurement (mode ends as "converged"
    or "max_samples").

    Returns:
        {
          "steady": list[result_dict],   # samples past TRANSIENT_DISCARD_N
          "mode": "converged" | "max_samples" | "verified",
          "saturated": bool,             # achieved rate < target → cap throttling
          "achieved_rate_hz": float,     # actual arrivals/s in steady window
          "target_rate_hz": float|None,
        }
    """
    from concurrent.futures import ThreadPoolExecutor
    samples_completed: list[tuple[int, dict]] = []  # (fired_idx, result)
    interval_s = (1.0 / target_rate_hz) if target_rate_hz else 0.0
    fired = 0
    # Pool size = concurrency (the hard cap). In-flight will never exceed c.
    pool_size = max(concurrency, 1)
    in_flight: list[tuple] = []  # (future, arrival_t_rel, q_depth, fired_idx)
    next_fire_t = time.time()
    samples_for_stability: list[float] = []  # decode_tps values, completion order
    stopping = False
    exit_mode = "max_samples"  # set to "converged" if CI triggers stop
    # Fire-rate tracking for saturation detection.
    fire_timestamps: list[float] = []  # times when arrivals actually fired
    # Audit ROBUSTNESS #5: bail out on persistent errors instead of
    # silently burning the whole MAX_SAMPLES_PER_LEVEL budget on failures.
    consecutive_errors = 0

    with ThreadPoolExecutor(max_workers=pool_size) as pool:
        while True:
            now = time.time()
            # Fire next arrival when due. For c=1 (target_rate_hz=None) fire
            # only when no request is in flight.
            # In-flight is hard-capped at `concurrency`. If at cap, the
            # metronome time can pass but we don't fire — we let next_fire_t
            # drift forward so that when a slot frees we can fire immediately
            # (catching up to the schedule, bounded by the cap).
            ready_to_fire = (
                not stopping
                and fired < MAX_SAMPLES_PER_LEVEL
                and len(in_flight) < concurrency
                and (
                    target_rate_hz is None  # c=1 baseline: fire-on-completion
                    or now >= next_fire_t
                )
            )
            if ready_to_fire:
                arrival_t_rel = now - cell_t0
                q_depth = peek_busy_slots(endpoint)
                fut = pool.submit(run_completion, endpoint, prompt, gen_size)
                in_flight.append((fut, arrival_t_rel, q_depth, fired))
                dashboard.update_in_flight(len(in_flight))
                fired += 1
                fire_timestamps.append(now)
                if target_rate_hz is not None:
                    next_fire_t += interval_s
                    # If next_fire_t is still in the past after the bump,
                    # we're behind schedule (cap was throttling). Don't
                    # burst-fire faster than once per loop iteration — the
                    # next iter will fire again immediately if conditions
                    # still hold. This is the saturation behavior.
                    if next_fire_t < now:
                        next_fire_t = now

            # Harvest completed futures.
            still_pending = []
            for fut, at_rel, qd, idx in in_flight:
                if not fut.done():
                    still_pending.append((fut, at_rel, qd, idx))
                    continue
                try:
                    result = fut.result()
                except Exception as e:
                    console.print(f"[red]request err c={concurrency} idx={idx}: {e}[/red]")
                    # Audit ROBUSTNESS #5: consistent server errors used
                    # to consume fire slots without ever advancing the
                    # stability or verify counters — burning all
                    # MAX_SAMPLES_PER_LEVEL arrivals on errors with no
                    # exit signal. Now we count consecutive failures and
                    # short-circuit the level if we hit the cap.
                    consecutive_errors += 1
                    if consecutive_errors >= ERROR_ABORT_THRESHOLD:
                        console.print(
                            f"[red]c={concurrency}: {consecutive_errors} consecutive "
                            f"request errors — aborting level early[/red]"
                        )
                        stopping = True
                        exit_mode = "errors"
                    continue
                consecutive_errors = 0  # any successful result resets the run
                is_steady = idx >= TRANSIENT_DISCARD_N
                row = _write_row(
                    raw_f, raw_lock, result,
                    preset=preset, context_size=context_size, gen_size=gen_size,
                    concurrency=concurrency, target_rate_hz=target_rate_hz,
                    arrival_wall_s=at_rel, queue_depth_at_arrival=qd,
                    fired_idx=idx, in_steady_state=is_steady,
                )
                dashboard.update(row=row)
                if result.get("decode_tps") is not None:
                    samples_completed.append((idx, result))
                    if is_steady:
                        samples_for_stability.append(result["decode_tps"])
            in_flight = still_pending
            dashboard.update_in_flight(len(in_flight))
            # Surface live sample progress to the header. Expected count
            # depends on which exit path is most likely first:
            #   verify mode active     → ~ TRANSIENT_DISCARD_N + VERIFY_WINDOW_N
            #   no verify (full mode)  → MAX_SAMPLES_PER_LEVEL upper bound
            if verify_prediction is not None:
                expected = TRANSIENT_DISCARD_N + VERIFY_WINDOW_N
                mode_hint = "verify"
            else:
                expected = MAX_SAMPLES_PER_LEVEL
                mode_hint = "stability"
            dashboard.update_progress(
                samples_done=len(samples_completed),
                samples_expected=expected,
                samples_mode=mode_hint,
            )

            # Verify-mode early exit: if we have a prediction and the
            # first VERIFY_WINDOW_N steady samples match it within
            # tolerance, exit immediately. Otherwise fall through to the
            # normal CI-based stability check.
            if (not stopping and verify_prediction is not None
                    and len(samples_for_stability) >= VERIFY_WINDOW_N):
                window = samples_for_stability[:VERIFY_WINDOW_N]
                mean = sum(window) / len(window)
                if verify_prediction > 0:
                    # Asymmetric tolerance — see VERIFY_TOLERANCE_* comment.
                    # Over-prediction (mean > pred) means the system is
                    # outperforming the model (c_sat>1 boost); wide tolerance.
                    # Under-prediction means unexpected degradation; tight
                    # tolerance triggers full measurement to investigate.
                    if mean >= verify_prediction:
                        deviation = (mean - verify_prediction) / verify_prediction
                        tolerance = VERIFY_TOLERANCE_OVER
                    else:
                        deviation = (verify_prediction - mean) / verify_prediction
                        tolerance = VERIFY_TOLERANCE_UNDER
                    if deviation <= tolerance:
                        stopping = True
                        exit_mode = "verified"
                    else:
                        # Outside tolerance — clear the prediction so we
                        # fall through to the standard stability path on
                        # this and future iterations.
                        verify_prediction = None

            # Stability check — once we have enough steady-state samples.
            if not stopping and len(samples_for_stability) >= STABILITY_WINDOW_N:
                window = samples_for_stability[-STABILITY_WINDOW_N:]
                if _ci_relative_halfwidth(window) < STABILITY_CI_REL_THRESHOLD:
                    stopping = True  # stop firing; still drain in-flight
                    exit_mode = "converged"

            # Hard cap on fired arrivals (failsafe).
            if fired >= MAX_SAMPLES_PER_LEVEL:
                stopping = True
                # exit_mode stays "max_samples" unless CI/verify already triggered

            # Exit when nothing left to fire and nothing in flight.
            if stopping and not in_flight:
                break

            # Sleep just long enough to avoid pinning the CPU but stay
            # responsive to fire times. For c=1 we can sleep longer since
            # we wait for completion anyway.
            sleep_s = min(0.02, max(0.001, interval_s / 4)) if interval_s else 0.05
            time.sleep(sleep_s)

    # Compute achieved arrival rate over the steady window to detect
    # saturation: if achieved << target, the cap was throttling fires.
    steady_fires = [t for t, idx in zip(fire_timestamps, range(len(fire_timestamps)))
                    if idx >= TRANSIENT_DISCARD_N]
    achieved_rate_hz: float = 0.0
    saturated = False
    if len(steady_fires) >= 2 and target_rate_hz is not None:
        span_s = steady_fires[-1] - steady_fires[0]
        if span_s > 0:
            achieved_rate_hz = (len(steady_fires) - 1) / span_s
        saturated = achieved_rate_hz < target_rate_hz * SATURATION_RATE_FRACTION

    return {
        "steady": [r for idx, r in samples_completed if idx >= TRANSIENT_DISCARD_N],
        "mode": exit_mode,
        "saturated": saturated,
        "achieved_rate_hz": achieved_rate_hz,
        "target_rate_hz": target_rate_hz,
    }


def run_ramp_cell(*, endpoint: str, prompt: str, gen_size: int,
                  preset: str, context_size: int, max_concurrency: int,
                  raw_f, raw_lock, dashboard, console: Console,
                  predict_shortcut: bool = True) -> dict:
    """Drive one (preset, ctx, gen) cell through a +1 ramp. The ramp climbs
    through every level up to max_concurrency, labeling knees as it goes
    (informational, not stop signals).

    Arrival rate at level c is FIXED: target_rate = c / T_req(c=1). In-flight
    is hard-capped at c, so the label "c=N" honestly means "at most N
    concurrent in-flight requests". Pre-saturation: open-loop at the target
    rate. At saturation: cap throttles fires → naturally becomes "fill
    slots as they free" (sustained-N closed-loop behavior). The hammer
    scenario only kicks in where it's honest — when the server actually
    can't keep up at the target rate.

    Stop conditions (any one):
      - aggregate t/s has exceeded c=1 baseline and then declined for
        AGGREGATE_DECLINE_N consecutive levels (definitively past peak,
        further levels just confirm the descent) — primary for c_sat>1
      - aggregate t/s has plateaued (within AGGREGATE_PLATEAU_TOL of
        peak for AGGREGATE_PLATEAU_N consecutive levels) — system has
        saturated, more samples produce no new information
      - per-req throughput drops below USELESS_DECODE_FRACTION of c=1
        baseline (backstop — catches pure c1/c systems eventually)
      - max_concurrency reached

    Labels (informational, never stop):
      - per-req knee: first c where per-req t/s < 0.5 × c=1 baseline
        (latency story — individual users start to feel it)
      - aggregate knee: first c where aggregate (per-req × c) stopped
        climbing vs the previous level (capacity story — adding more
        concurrent users stops increasing total work done)
      - saturated_from_c: first c where achieved fire rate fell below
        SATURATION_RATE_FRACTION × target (cap is throttling — sustained
        throughput at this c is what we're measuring)"""
    cell_t0 = time.time()

    dashboard.update(current={
        "preset": preset, "context_size": context_size, "gen_size": gen_size,
        "concurrency": 1, "target_rate_hz": None,
    })
    dashboard.log_event(
        f"→ cell {preset} ctx={context_size} gen={gen_size}: c=1 baseline…",
        level="transition", type="cell_start",
        preset=preset, context_size=context_size, gen_size=gen_size,
    )

    lvl = _run_level(
        endpoint=endpoint, prompt=prompt, gen_size=gen_size,
        preset=preset, context_size=context_size, concurrency=1,
        target_rate_hz=None, cell_t0=cell_t0,
        raw_f=raw_f, raw_lock=raw_lock,
        dashboard=dashboard, console=console,
    )
    c1_steady = lvl["steady"]
    if not c1_steady:
        dashboard.log_event(
            f"cell {preset} ctx={context_size} gen={gen_size}: "
            f"c=1 baseline produced no steady samples — skipping ramp",
            level="error", type="cell_skip",
            preset=preset, context_size=context_size, gen_size=gen_size,
        )
        return {"preset": preset, "context_size": context_size,
                "gen_size": gen_size, "c1_decode_tps": None,
                "t_req_s": None, "stop_reason": "no_baseline",
                "knee_per_req_c": None, "knee_aggregate_c": None,
                "levels": [{"c": 1, "mode": lvl["mode"]}]}

    # Token-weighted per-stream decode for the c=1 baseline (matches the
    # c>=2 path at line 1990 and bench_data.aggregate_ramp_by_cell).
    # Arithmetic mean here would silently disagree with the post-run
    # table — same class of bug as #23 but at the c=1 path that #23
    # missed. The audit caught it.
    c1_toks_dt = [(s["predicted_n"], s["predicted_n"] / s["decode_tps"])
                  for s in c1_steady
                  if s.get("predicted_n") and s.get("decode_tps")]
    if c1_toks_dt:
        c1_toks_sum = sum(t for t, _ in c1_toks_dt)
        c1_dec_time_sum = sum(d for _, d in c1_toks_dt)
        c1_decode_tps = (c1_toks_sum / c1_dec_time_sum
                         if c1_dec_time_sum > 0 else 0.0)
    else:
        c1_decode_tps = sum(s["decode_tps"] for s in c1_steady) / len(c1_steady)
    # T_req: median rather than arithmetic mean — drives target_rate_hz
    # at all higher c levels (target = c / c1_wall_s). Short-EOS samples
    # at c=1 would pull the mean down and inflate all subsequent targets,
    # cascading through verify-mode predictions and saturation labels.
    from statistics import median as _median
    c1_wall_s = _median(s["wall_s"] for s in c1_steady)
    c1_aggregate_tps = c1_decode_tps  # by definition at c=1
    dashboard.log_event(
        f"c=1 baseline ({lvl['mode']}): {c1_decode_tps:.1f} t/s per req, "
        f"T_req={c1_wall_s:.2f}s ({len(c1_steady)} steady samples)",
        level="info", type="baseline_complete",
        c=1, per_req_decode_tps=c1_decode_tps, t_req_s=c1_wall_s,
        n_steady=len(c1_steady), mode=lvl["mode"],
    )

    levels: list[dict] = [{
        "c": 1, "mode": lvl["mode"],
        "per_req_decode_tps": c1_decode_tps,
        "aggregate_decode_tps": c1_aggregate_tps,
        "wall_mean_s": c1_wall_s,
        "target_rate_hz": None,
    }]
    knee_per_req_c: int | None = None
    knee_aggregate_c: int | None = None
    saturated_from_c: int | None = None
    stop_reason = "max_concurrency"

    # Aggregate-decline + plateau termination state (see AGGREGATE_* comments).
    peak_aggregate = c1_aggregate_tps
    peak_aggregate_c = 1
    passed_c1_aggregate = False
    aggregate_decline_count = 0
    aggregate_plateau_count = 0

    for c in range(2, max_concurrency + 1):
        # Fixed: target arrival rate based on c=1 baseline. "c=N" honestly
        # means "N-user equivalent sustained load." See run_ramp_cell
        # docstring for why adaptive rate was rejected.
        target_rate_hz = c / c1_wall_s

        # Prediction shortcut: once MIN_FULL_LEVELS levels are measured
        # fully, fit per_req(c) and run subsequent levels in verify mode.
        # Skip-out when measurement matches prediction within tolerance.
        verify_pred = None
        if predict_shortcut and len(levels) >= MIN_FULL_LEVELS:
            fit = _fit_per_req_model(levels)
            if fit is not None:
                verify_pred = _predict_per_req(c, fit)

        dashboard.update(current={
            "preset": preset, "context_size": context_size, "gen_size": gen_size,
            "concurrency": c, "target_rate_hz": target_rate_hz,
        })
        if verify_pred is not None:
            dashboard.log_event(
                f"→ c={c} @ {target_rate_hz:.2f} Hz (verify mode, "
                f"predicted {verify_pred:.1f} t/s)…",
                level="info", type="level_start",
                c=c, target_rate_hz=target_rate_hz,
                verify_prediction=verify_pred,
            )
        else:
            dashboard.log_event(
                f"→ c={c} @ {target_rate_hz:.2f} Hz…",
                level="info", type="level_start",
                c=c, target_rate_hz=target_rate_hz,
            )

        lvl = _run_level(
            endpoint=endpoint, prompt=prompt, gen_size=gen_size,
            preset=preset, context_size=context_size, concurrency=c,
            target_rate_hz=target_rate_hz, cell_t0=cell_t0,
            raw_f=raw_f, raw_lock=raw_lock,
            dashboard=dashboard, console=console,
            verify_prediction=verify_pred,
        )
        steady = lvl["steady"]
        if not steady:
            dashboard.log_event(
                f"c={c}: no steady samples ({lvl['mode']}) — "
                f"recording and continuing",
                level="warn", type="no_steady",
                c=c, mode=lvl["mode"],
            )
            levels.append({"c": c, "mode": lvl["mode"],
                           "per_req_decode_tps": None,
                           "aggregate_decode_tps": None,
                           "wall_mean_s": None,
                           "target_rate_hz": target_rate_hz,
                           "achieved_rate_hz": lvl.get("achieved_rate_hz"),
                           "saturated": lvl.get("saturated", False)})
            continue

        # Token-weighted per-stream decode rate (matches the post-run
        # aggregator in bench_data.aggregate_ramp_by_cell). Using
        # arithmetic mean here would silently disagree with the table
        # in summary.md — observed 2026-06-06 Ada run where the cell
        # summary peak@c said c=6 but the per-cell table showed peak
        # at c=4 (the token-weighted truth).
        toks_dt = [(s["predicted_n"], s["predicted_n"] / s["decode_tps"])
                   for s in steady
                   if s.get("predicted_n") and s.get("decode_tps")]
        if toks_dt:
            toks_sum = sum(t for t, _ in toks_dt)
            time_sum = sum(d for _, d in toks_dt)
            per_req = toks_sum / time_sum if time_sum > 0 else 0.0
        else:
            per_req = sum(s["decode_tps"] for s in steady) / len(steady)
        wall_mean = sum(s["wall_s"] for s in steady) / len(steady)
        aggregate = per_req * c
        levels.append({
            "c": c, "mode": lvl["mode"],
            "per_req_decode_tps": per_req,
            "aggregate_decode_tps": aggregate,
            "wall_mean_s": wall_mean,
            "target_rate_hz": target_rate_hz,
            "achieved_rate_hz": lvl.get("achieved_rate_hz"),
            "saturated": lvl.get("saturated", False),
        })
        if lvl.get("saturated") and saturated_from_c is None:
            saturated_from_c = c
            ach = lvl.get("achieved_rate_hz", 0.0)
            # Only flag saturation when it happens past c=2 — see #24.
            if c > 2:
                dashboard.log_event(
                    f"saturated from c={c} (achieved {ach:.2f} Hz vs "
                    f"target {target_rate_hz:.2f} Hz)",
                    level="transition", type="saturation_label",
                    c=c, achieved_rate_hz=ach, target_rate_hz=target_rate_hz,
                )
        prev_agg = levels[-2].get("aggregate_decode_tps")
        per_req_frac = per_req / c1_decode_tps if c1_decode_tps else 0.0
        agg_frac = (aggregate / prev_agg) if prev_agg else 1.0
        # Level summary — color hints "peak" when aggregate exceeded prior peak.
        is_new_peak = aggregate > peak_aggregate
        level_color = "peak" if is_new_peak else "info"
        dashboard.log_event(
            f"c={c} ({lvl['mode']}): per-req {per_req:.1f} t/s "
            f"({per_req_frac*100:.0f}% of c=1), aggregate {aggregate:.1f} t/s "
            f"({agg_frac*100:.0f}% of c={c-1})",
            level=level_color, type="level_complete",
            c=c, mode=lvl["mode"],
            per_req_decode_tps=per_req, aggregate_decode_tps=aggregate,
            per_req_frac_of_c1=per_req_frac, agg_frac_of_prev=agg_frac,
        )

        # Label per-req knee inline. Aggregate peak is computed after the
        # ramp ends using argmax over all levels — labeling on the first
        # downtick fires on noise (e.g. c=2 < c=1 by 2% within a cell that
        # keeps climbing past c=1 at c=3+).
        if knee_per_req_c is None and per_req_frac < 0.5:
            knee_per_req_c = c
            dashboard.log_event(
                f"per-req knee at c={c} (decode {per_req_frac*100:.0f}% of c=1)",
                level="transition", type="knee_per_req",
                c=c, per_req_frac=per_req_frac,
            )

        # Update aggregate-decline tracking (see AGGREGATE_DECLINE_N comment).
        # passed_c1_aggregate latches True once aggregate exceeds the c=1
        # baseline — that proves we've found a c_sat>1 bounce. Without it
        # latching, we'd start counting declines in the c=2..c_sat dip
        # region where per-req c1/c overhead pulls aggregate below c=1
        # before the bounce kicks in.
        if aggregate > peak_aggregate:
            if aggregate > c1_aggregate_tps:
                passed_c1_aggregate = True
            peak_aggregate = aggregate
            peak_aggregate_c = c
            aggregate_decline_count = 0
            aggregate_plateau_count = 0
        elif passed_c1_aggregate and c >= AGGREGATE_DECLINE_MIN_C:
            # Within tolerance of peak = plateau; clearly below = decline.
            # Both are termination signals but they tell different stories.
            within_plateau = (
                aggregate >= peak_aggregate * (1 - AGGREGATE_PLATEAU_TOL)
            )
            if within_plateau:
                aggregate_plateau_count += 1
                # Plateau levels aren't really "declines" — reset decline
                # count so a true decline (below tolerance) doesn't get
                # confused with hovering near peak.
                aggregate_decline_count = 0
            else:
                aggregate_decline_count += 1
                # Decline means we've left the peak neighborhood — reset
                # plateau count so future re-entries to the peak band
                # start fresh.
                aggregate_plateau_count = 0
            if aggregate_decline_count >= AGGREGATE_DECLINE_N:
                stop_reason = "aggregate_decline"
                dashboard.log_event(
                    f"stop: aggregate t/s declined {aggregate_decline_count} "
                    f"consecutive levels past peak ({peak_aggregate:.1f} t/s "
                    f"at c={peak_aggregate_c}) — past useful concurrency",
                    level="warn", type="termination", stop_reason=stop_reason,
                    peak=peak_aggregate, peak_c=peak_aggregate_c,
                )
                break
            if aggregate_plateau_count >= AGGREGATE_PLATEAU_N:
                stop_reason = "aggregate_plateau"
                dashboard.log_event(
                    f"stop: aggregate t/s within "
                    f"{int(AGGREGATE_PLATEAU_TOL*100)}% of peak "
                    f"({peak_aggregate:.1f} t/s at c={peak_aggregate_c}) for "
                    f"{aggregate_plateau_count} consecutive levels — "
                    f"no new information",
                    level="warn", type="termination", stop_reason=stop_reason,
                    peak=peak_aggregate, peak_c=peak_aggregate_c,
                )
                break

        # Backstop: per-req has degraded so badly that even at this c (with
        # the in-flight cap protecting us from runaway queue) no user would
        # tolerate the latency. Catches pure c1/c systems where the
        # aggregate-decline check above never fires (aggregate never beats
        # c=1, so passed_c1_aggregate stays False).
        if per_req_frac < USELESS_DECODE_FRACTION:
            stop_reason = "useless_per_req"
            dashboard.log_event(
                f"stop: per-req {per_req_frac*100:.0f}% of c=1 "
                f"< {USELESS_DECODE_FRACTION*100:.0f}% useless floor",
                level="warn", type="termination", stop_reason=stop_reason,
                per_req_frac=per_req_frac,
            )
            break

    # Aggregate peak = argmax over all measured levels. Computed at the end
    # so we don't get fooled by a single down-tick within noise.
    valid = [lvl for lvl in levels if lvl.get("aggregate_decode_tps") is not None]
    if valid:
        peak_lvl = max(valid, key=lambda lvl: lvl["aggregate_decode_tps"])
        knee_aggregate_c = peak_lvl["c"]

    # Always emit a termination event, even when stop_reason is the
    # "ran to max_concurrency" fall-through case (which doesn't break
    # out of the loop and so didn't trigger one of the inline log_events
    # above). Audit #10: weatherman's decision-diff couldn't distinguish
    # "completed at max_c cap" from "cell never ran" without this.
    if stop_reason not in ("aggregate_decline", "aggregate_plateau",
                            "useless_per_req", "no_baseline"):
        peak_val = peak_lvl["aggregate_decode_tps"] if valid else None
        peak_c_str = f" at c={knee_aggregate_c}" if knee_aggregate_c else ""
        dashboard.log_event(
            f"cell complete (stop={stop_reason}): "
            f"aggregate peak {peak_val:.1f} t/s{peak_c_str}"
            if peak_val is not None
            else f"cell complete (stop={stop_reason})",
            level="success", type="termination", stop_reason=stop_reason,
            peak=peak_val, peak_c=knee_aggregate_c,
        )

    return {
        "preset": preset, "context_size": context_size, "gen_size": gen_size,
        "c1_decode_tps": c1_decode_tps, "t_req_s": c1_wall_s,
        "stop_reason": stop_reason,
        "knee_per_req_c": knee_per_req_c,
        "knee_aggregate_c": knee_aggregate_c,
        "saturated_from_c": saturated_from_c,
        "levels": levels,
    }


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
    for note_key in ("parallel_override", "ctx_size_override", "kv_unified_added"):
        if server_notes.get(note_key):
            console.print(f"[yellow]{server_notes[note_key]}[/yellow]")
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
    # Bind these before the try so the finally can always clean them up
    # — even if construction inside the try raises. Audit CRITICAL #2
    # caught a missing raw_f close on non-KeyboardInterrupt exceptions.
    raw_f = None
    dashboard = None
    run_failed = False
    try:
        if server_cfg.get("launch"):
            proc = launch_server(argv, env_overrides, run_dir / "server.log", console)
            with console.status("[bold cyan]waiting for /health…[/bold cyan]"):
                wait_for_health(endpoint, server_cfg.get("health_timeout_s", 120), console)
            console.print("[green]server ready[/green]")
            # Parse the server's own startup log for VRAM allocation truth.
            # If any weights ended up on CPU, the benchmark will be
            # meaningfully slower than expected — warn loudly so the user
            # can abort before burning hours on a misconfigured run.
            # MoE detection: if -cmoe / --n-cpu-moe is explicitly set,
            # Host allocations are intentional and the warning is a
            # false positive — pass that signal to the checker.
            moe_intentional = _has_moe_offload(
                (server_cfg.get("args") or [])
            )
            check_vram_budget(run_dir / "server.log", console,
                              moe_offload_intentional=moe_intentional)
        else:
            wait_for_health(endpoint, 5.0, console)
            # When attached to an existing server, confirm its --parallel
            # slot count is enough for the ramp's max_concurrency cap.
            required_parallel = bench_data.get_max_concurrency(
                cfg.get("sweep", {}) or {})
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
                                f"but ramp's max_concurrency is {required_parallel}. Requests "
                                f"above the slot count will queue — TTFT and aggregate "
                                f"throughput will be misleading. Restart llama-server with "
                                f"--parallel {required_parallel}.[/yellow]"
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
        max_concurrency: int = bench_data.get_max_concurrency(sweep)
        predict_shortcut: bool = bool(sweep.get("predict_shortcut", True))
        # Migration warnings for the closed-loop config knobs. These were
        # replaced by the rolling ramp in 2026-06-04 — see
        # project_design_decisions / project_active_pin memories.
        for legacy in ("concurrency_levels", "rounds", "warmup_rounds",
                       "gate_between_concurrency", "gate_only_on_decrease",
                       "gate_timeout_s"):
            if legacy in sweep:
                console.print(
                    f"[yellow]config: '{legacy}' is no longer used — "
                    f"the rolling ramp replaces closed-loop sweep semantics. "
                    f"Use 'max_concurrency' to cap the ramp.[/yellow]"
                )
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

        # 5. sweep — preset × ctx × gen, each cell ramps c=1..max_concurrency
        # with stability-driven sample sizing and knee-detection abort. No
        # explicit warmup pass — the c=1 baseline at each cell subsumes it.
        total_cells = len(preset_names) * len(context_sizes) * len(gen_sizes)
        # Pessimistic upper bound for the dashboard's progress fraction —
        # better to over-estimate (bar climbs steadily, run finishes
        # "under expected") than under-estimate (bar pins at 100% and
        # keeps incrementing, looks like the run is overrunning). Worst
        # case: every cell ramps all the way to max_concurrency and every
        # level uses MAX_SAMPLES_PER_LEVEL. Most real cells use far fewer
        # samples (verify mode + early termination), so this typically
        # over-estimates 2-4x and the run lands under it.
        total_requests_est = (total_cells * max_concurrency
                              * MAX_SAMPLES_PER_LEVEL)
        raw_path = run_dir / "raw.jsonl"
        raw_f = raw_path.open("w")
        raw_lock = threading.Lock()
        live_mode = (cfg.get("output") or {}).get("live", True) and not args.no_live

        dashboard = Dashboard(console, total_requests_est, manifest, cfg, telemetry)
        dashboard.attach_events_log(run_dir / "events.log")

        # Diagnostic /slots probe at startup so we can verify the busy-check
        # logic against this server version.
        if max_concurrency > 1:
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
                    "[yellow]/slots endpoint unavailable — queue_depth_at_arrival "
                    "will be recorded as 0. Ensure llama-server has --slots enabled.[/yellow]"
                )

        # Closed-loop run_batch / ConcurrencySampler / gate_* logic was
        # removed 2026-06-04 when the rolling ramp replaced discrete-c sweeps.
        # The ramp driver lives in run_ramp_cell().

        def iter_cells():
            for preset in preset_names:
                for cs in context_sizes:
                    for gs in gen_sizes:
                        yield preset, cs, gs

        cell_summaries: list[dict] = []

        def run_one_cell(preset: str, cs: int, gs: int) -> None:
            summary = run_ramp_cell(
                endpoint=endpoint,
                prompt=sized_prompts[(preset, cs)],
                gen_size=gs, preset=preset, context_size=cs,
                max_concurrency=max_concurrency,
                raw_f=raw_f, raw_lock=raw_lock,
                dashboard=dashboard, console=console,
                predict_shortcut=predict_shortcut,
            )
            cell_summaries.append(summary)

        if live_mode:
            with Live(dashboard.layout, console=console, refresh_per_second=8,
                      screen=False):
                dashboard.update()  # initial paint
                dashboard.start_ticker(0.25)
                try:
                    for preset, cs, gs in iter_cells():
                        run_one_cell(preset, cs, gs)
                finally:
                    dashboard.stop_ticker()
        else:
            with Progress(TextColumn("[progress.description]{task.description}"),
                          BarColumn(), TextColumn("{task.completed}/{task.total}"),
                          TimeElapsedColumn(), console=console) as prog:
                tid = prog.add_task("ramp cells", total=total_cells)
                for preset, cs, gs in iter_cells():
                    run_one_cell(preset, cs, gs)
                    prog.update(tid, advance=1,
                                description=f"{preset} ctx={cs} g={gs}")
        if telemetry:
            telemetry.stop()

        # Cell summaries: cheap text recap of where each cell ended.
        for s in cell_summaries:
            c1 = s.get("c1_decode_tps")
            if c1 is None:
                console.print(
                    f"[red]cell {s['preset']} ctx={s['context_size']} "
                    f"gen={s['gen_size']}: no baseline[/red]"
                )
                continue
            t_req = s.get("t_req_s")
            cs_run = [lvl["c"] for lvl in s.get("levels", [])]
            agg_peak = s.get("knee_aggregate_c")
            per_req_knee = s.get("knee_per_req_c")
            sat_from = s.get("saturated_from_c")
            stop = s.get("stop_reason", "?")
            # Saturation@c=2 is the trivial case for any memory-bound
            # system (target rate = c/T_req(c=1) is never sustainable at
            # c≥2 if T_req grows with c, i.e. always). Only surface when
            # it's >2 — that means the system can actually sustain target
            # rate past c=1, which is meaningful information.
            sat_str = (f"saturated@c={sat_from} · "
                       if sat_from is not None and sat_from > 2 else "")
            console.print(
                f"[dim]cell {s['preset']} ctx={s['context_size']} "
                f"gen={s['gen_size']}: c=1={c1:.1f} t/s T_req={t_req:.2f}s · "
                f"levels {cs_run} · "
                f"per-req knee@c={per_req_knee or '—'} · "
                f"aggregate peak@c={agg_peak or '—'} · "
                f"{sat_str}"
                f"stop={stop}[/dim]"
            )

        write_summary(run_dir)
        _print_summary_table(console, run_dir)

    except KeyboardInterrupt:
        run_failed = True
        console.print("[yellow]interrupted[/yellow]")
    except Exception as e:
        run_failed = True
        # Surface the actual failure rather than letting it propagate
        # silently while finally cleans up — audit CRITICAL #3 caught
        # the old behavior where any non-KeyboardInterrupt exception
        # still ended with "done." giving the false impression of
        # success.
        console.print(f"[bold red]run failed: {type(e).__name__}: {e}[/bold red]")
        import traceback as _tb
        console.print(f"[dim]{_tb.format_exc()}[/dim]")
    finally:
        # Always-cleanup, in order from most-dependent to least: stop
        # ticker (if dashboard reached construction), close events.log,
        # close raw.jsonl (most important — preserves whatever data we
        # got even if the run died mid-cell), terminate server.
        if raw_f is not None:
            try:
                raw_f.close()
            except OSError:
                pass
        if dashboard is not None:
            dashboard.close_events_log()
        if proc is not None:
            terminate_server(proc)
            console.print("[dim]server terminated[/dim]")

    # 7. auto-plot — even on failure, partial raw.jsonl is worth plotting.
    if not args.no_plot:
        try:
            subprocess.run([sys.executable, "plot.py", str(run_dir)], check=False)
        except Exception as e:
            console.print(f"[red]plot.py failed: {e}[/red]")

    if run_failed:
        console.print(f"\n[bold red]run did not complete cleanly.[/bold red] "
                      f"partial results in {run_dir}")
        return 1
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
    presets = bench_data.get_prompt_presets(sweep)
    max_c = bench_data.get_max_concurrency(sweep)
    n_cells = len(ctx) * len(gs) * max(1, len(presets))

    g = Table.grid(padding=(0, 1))
    g.add_column(style="bold cyan", no_wrap=True)
    g.add_column(overflow="fold")
    g.add_row("model", str(model) + (f"  [dim]draft={draft}[/dim]" if draft else ""))
    g.add_row("flags", flags or "[dim](none)[/dim]")
    g.add_row("sweep",
              f"ctx={ctx} × gen={gs}  presets={presets}  "
              f"max_c={max_c}  [dim]({n_cells} ramp cells)[/dim]")
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
    def _gib(mib):
        return f"{mib/1024:.0f}" if isinstance(mib, (int, float)) else "?"

    def _opt(v, suffix=""):
        return f"{v}{suffix}" if v is not None else "?"

    for g in m.get("gpus", []) or []:
        in_use = "✓" if (not sel_set or str(g.get("index")) in sel_set) else " "
        vram = f"{_gib(g.get('vram_free_mib'))}/{_gib(g.get('vram_total_mib'))}G"
        pcie_gen, pcie_w = g.get("pcie_gen"), g.get("pcie_width")
        pcie = f"g{pcie_gen}x{pcie_w}" if pcie_gen is not None and pcie_w is not None else "?"
        gpu_tbl.add_row(
            str(g.get("index")),
            _short_gpu_name(g.get("name")),
            vram,
            str(g.get("driver") or "?"),
            _opt(g.get("power_limit_w"), "W"),
            pcie,
            in_use,
        )
    console.print(Panel(Group(head, Text(""), gpu_tbl), title="system manifest",
                        border_style="magenta", padding=(0, 1)))


def _print_summary_table(console: Console, run_dir: Path) -> None:
    """Print the rolling-ramp summary — one table per cell showing the c
    ramp with the hyperbolic fit overlay (predicted t/s + deviation %)."""
    rows = bench_data.load_jsonl(run_dir / "raw.jsonl")
    if not rows:
        return
    cells = bench_data.aggregate_ramp_by_cell(rows)
    if not cells:
        return

    for key in sorted(cells.keys()):
        preset, cs, gs = key
        cell = cells[key]
        c1 = cell.get("c1_decode_tps")
        fit = cell.get("fit")
        peak_c = cell.get("aggregate_peak_c")
        thresh_c = cell.get("per_req_threshold_c")
        title = (f"ramp — preset={preset} ctx={cs} gen={gs}"
                 f"  ·  c1={c1:.1f} t/s" if c1 else
                 f"ramp — preset={preset} ctx={cs} gen={gs} (no baseline)")
        if c1:
            if fit is not None:
                title += (f" · fit ceiling≈{fit['aggregate_ceiling']:.0f} t/s "
                          f"(T_fixed={fit['t_fixed']*1000:.1f}ms, "
                          f"T_per_stream={fit['t_per_stream']*1000:.1f}ms)")
            if peak_c is not None:
                title += f" · agg peak@c={peak_c}"
            if thresh_c is not None:
                title += f" · <50% c=1@c={thresh_c}"

        t = Table(title=title,
                  caption="[dim]per-req token-weighted from steady samples; "
                          "Δ = measured vs hyperbolic fit prediction[/dim]")
        t.add_column("c", justify="right")
        t.add_column("per-req t/s", justify="right", style="yellow")
        t.add_column("predicted t/s", justify="right", style="dim")
        t.add_column("Δ", justify="right")
        t.add_column("aggregate t/s", justify="right", style="bold bright_cyan")
        t.add_column("prefill t/s", justify="right")
        t.add_column("TTFT ms", justify="right")
        t.add_column("accept %", justify="right")
        t.add_column("n", justify="right")
        for lvl in cell["levels"]:
            per_req = lvl.get("per_req_decode_tps")
            pred = lvl.get("predicted_per_req")
            dev = lvl.get("deviation")
            agg = lvl.get("aggregate_decode_tps")
            ttft_m = lvl.get("ttft_ms_mean")
            ttft_p = lvl.get("ttft_ms_p95")
            ttft_str = (f"{ttft_m:.0f} (p95 {ttft_p:.0f})"
                        if ttft_m is not None and ttft_p is not None else "-")
            dev_str = (f"{dev*100:+.1f}%" if dev is not None else "-")
            t.add_row(
                str(lvl["c"]),
                f"{per_req:.1f}" if per_req is not None else "-",
                f"{pred:.1f}" if pred is not None else "-",
                dev_str,
                f"{agg:.1f}" if agg is not None else "-",
                f"{lvl['prefill_tps_tw']:.0f}" if lvl.get("prefill_tps_tw") else "-",
                ttft_str,
                f"{lvl['accept_mean']*100:.1f}" if lvl.get("accept_mean") is not None else "-",
                str(lvl.get("n_steady", 0)),
            )
        console.print(t)

    if len(cells) > 1:
        console.print(f"[dim]full breakdown in {run_dir}/plots/summary.md[/dim]")


def _build_cell_table(cells: dict, title: str) -> Table:
    """Render a per-(ctx, gen) cell table — token-weighted means + std-dev.

    Decode and prefill columns show `mean ±σ` so high variance is visible
    at a glance. High σ% (e.g. >10%) signals noisy data — thermal drift,
    transient server churn, or insufficient rounds for replication.
    """
    t = Table(title=title,
              caption="[dim]rates token-weighted; ± is per-request stddev[/dim]")
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
        prefill_mean = c.get("prefill_mean")
        prefill_std = c.get("prefill_std", 0.0)
        prefill_str = (f"{prefill_mean:.1f} ±{prefill_std:.0f}"
                       if prefill_mean is not None else "-")
        decode_mean = c.get("decode_mean")
        decode_std = c.get("decode_std", 0.0)
        decode_str = (f"{decode_mean:.1f} ±{decode_std:.1f}"
                      if decode_mean is not None else "-")
        ttft_mean = c.get("ttft_ms_mean")
        ttft_std = c.get("ttft_ms_std", 0.0)
        ttft_str = (f"{ttft_mean:.0f} ±{ttft_std:.0f}"
                    if ttft_mean is not None else "-")
        t.add_row(
            str(cs), str(gs),
            prefill_str,
            decode_str,
            f"{c['decode_ms_per_token']:.2f}" if c.get("decode_ms_per_token") is not None else "-",
            ttft_str,
            f"{c['accept_mean']*100:.1f}" if c.get("accept_mean") is not None else "-",
            str(c.get("n", 0)),
        )
    return t


if __name__ == "__main__":
    sys.exit(main())
