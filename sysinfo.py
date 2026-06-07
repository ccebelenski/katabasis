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
    # Default to llama-server on $PATH when no explicit binary is given.
    # Resolve via shutil.which so the recorded manifest captures the actual
    # binary used, not the literal name "llama-server".
    if not binary:
        resolved = shutil.which("llama-server")
        if resolved:
            binary = resolved
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
