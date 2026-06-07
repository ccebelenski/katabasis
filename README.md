# katabasis

**A drop-in replacement for `llama-bench` that measures across the regimes your model actually sees.**

`kata` (short for *katabasis* — Greek for *descent*) is a benchmark suite for [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`. It does what `llama-bench` does — prefill/decode throughput, draft/MTP acceptance, sweeps across context and generation size — but produces numbers you can actually trust and act on, because it measures across a *matrix* of workloads instead of a single point.

A single `tokens/sec` number is almost always misleading. The same model on the same GPU can show 100 t/s at ctx=256 and 40 t/s at ctx=16384 — *and the second number is the one that matters for chat sessions, RAG, and code assistants*. Kata sweeps prompt preset × context size × generation size automatically, captures TTFT and per-request decode (not just aggregate throughput), and produces honest, scrutiny-ready artifacts you can paste straight into a blog post or PR comment.

```
bench.yaml ──► kata.py ──► live dashboard + runs/<ts>__<rig>/
                                            ├─ raw.jsonl       (every request)
                                            ├─ events.log      (every decision)
                                            ├─ summary.md      (headline + per-cell tables)
                                            ├─ summary.json    (machine-readable)
                                            ├─ system.json     (rig + llama.cpp commit)
                                            ├─ server.log      (full server output)
                                            ├─ gpu_telemetry.csv
                                            ├─ narrative.md    (on demand)
                                            └─ plots/*.png
                                                  │
                          weatherman.py (TUI) ◄───┘
```

---

## Why kata instead of llama-bench

`llama-bench` is the standard, and it's fine — but it gives you one prefill number and one decode number per configuration. Real serving workloads aren't one number. Context size dominates everything; a model that decodes at 100 t/s with an empty context might be at 40 t/s once the conversation has scrolled, and *only kata tells you both numbers in one run*.

| concern | `llama-bench` | `kata` |
|---|---|---|
| single-stream prefill / decode t/s | ✓ | ✓ |
| TTFT (client-measured, not server-estimated) | — | ✓ |
| sweep across context size in one run | partial | ✓ |
| sweep across prompt shape (niah vs code vs chat) | — | ✓ |
| draft / MTP speculative-decode acceptance metrics | partial | ✓ |
| token-weighted aggregation (honest under load mixing) | — | ✓ |
| VRAM / CPU-spillover detection (refuses bad runs) | — | ✓ |
| persistent, scrutiny-ready artifacts (`raw.jsonl`, `events.log`) | — | ✓ |
| post-run interactive review TUI (`weatherman`) | — | ✓ |
| cross-rig A/B with set-intersection + ctx-interpolation | — | ✓ |
| optional concurrency ramp (operator verdict per cell) | — | ✓ |

Kata isn't competing with llama-bench for its niche (the in-tree quick check). It's the next tool you reach for once you want numbers that survive expert scrutiny.

---

## Philosophy

- **Sweep first, single-number last.** A benchmark that returns one number for a model that's used across 8x context-size variation is misleading by construction. The matrix *is* the answer.

- **Methodology has to defend itself.** Token-weighted aggregation, hyperbolic two-parameter fit, asymmetric verify tolerance — each choice is documented inline with the rationale. If a reviewer challenges any number, the answer is in the code, not in a vibe.

- **Fail loud, not silently bad.** `--fit off` is enforced policy; the startup check parses the server log for `-fit on` spillover, mismatched `cuda_visible_devices`, OOM, and partial layer offload — all surfaced as red warnings, not buried in logs. Better to refuse a misconfigured run than waste hours on bad data.

- **The whole stack matters, not just the silicon.** Kata measures the *hardware × llama.cpp build × config* combination. NCCL state, P2P availability, KV quantization, slot-prefill serialization — all are part of "what this run measured." `system.json` captures the relevant context.

- **Artifacts are designed to be posted.** `summary.md`, `events.log`, and the plot PNGs are meant to live in PR comments, blog posts, and Reddit threads. Raw data is git-committable so anyone can re-analyze.

---

## Quick start

### Install

Python 3.10+. Per-host venv (kata is `/mnt/ssd`-friendly across mounted hosts):

```bash
git clone https://github.com/<you>/kata
cd kata
python3 -m venv venv-$(hostname -s)
source venv-$(hostname -s)/bin/activate
pip install -r requirements.txt
```

You'll need `llama-server` on `$PATH`. Build llama.cpp from source with CUDA support, or use your distro's package if it ships one.

`nvidia-smi` / `rocm-smi` are used for GPU manifest and live telemetry. Missing tools degrade gracefully.

### Run a benchmark

```bash
python kata.py configs/qwen36-IQ4_NL.yaml
```

That's it. Kata launches the server, drives the sweep across every (preset, ctx, gen) cell, shuts the server down, and writes `runs/<ts>__<rig>/` with all artifacts.

Wall time scales with cell count. A typical 6-cell single-stream sweep finishes in 5–15 minutes; a full sweep with concurrency ramp can run 1–3 hours.

### Review the results

Read `runs/<ts>__<rig>/summary.md` directly:

```markdown
## Headline

| cell                       | decode t/s | prefill t/s | TTFT (median) |
|----------------------------|-----------:|------------:|--------------:|
| `code` ctx=2048 gen=512    |      107.2 |        4820 |          412ms|
| `code` ctx=8192 gen=512    |       98.5 |        4710 |         1.62s |
| `code` ctx=16384 gen=512   |       93.6 |        4640 |         3.41s |
| `niah` ctx=16384 gen=512   |       94.5 |        4655 |         3.39s |
```

The same model, same GPU, same llama.cpp build — a 14% spread in decode and an 8x spread in TTFT depending on context. Numbers an operator can actually plan around.

Browse interactively with the TUI:

```bash
python weatherman.py runs/
```

- **Summary tab** — per-run headline, per-cell tables
- **Charts tab** — throughput, prefill, TTFT, draft-accept curves
- **Events tab** — chronological decision log
- **A/B tab** — pin two runs (`space`), press `c` — coverage banner + decision diff + cross-ctx interpolated comparison + system diff
- **Search tab** — query events across the entire run tree
- Press `n` to export the current run as a markdown narrative

---

## What kata measures

For each `(preset, context_size, gen_size)` cell, kata measures:

- **Decode throughput** — token-weighted (sum_tokens / sum_time across requests), not arithmetic mean of per-request rates. This is the difference between an honest number and a flattering one when request sizes vary.
- **Prefill throughput** — likewise token-weighted; computed from server-reported prefill spans.
- **TTFT** — client-measured time-to-first-token via the SSE stream, reported as median + IQR (robust to short-EOS outliers). This is what users actually feel; server-side estimates often understate it.
- **Draft / MTP acceptance** — when speculative decode is enabled, kata captures draft accept rate per cell. Lets you tell whether MTP is paying off on this workload.
- **Spread, not just averages** — per-request distribution is preserved in `raw.jsonl`; the summary shows median + range so you can see when a cell is bimodal.

**Prompt presets** are workload shapes, not just sizes:

- `niah` — needle-in-a-haystack style long-context retrieval (lorem ipsum body + buried question)
- `code` — realistic code-completion prompt (function bodies, imports, continuation)
- `chat` — multi-turn conversational scaffold

Different prompt shapes hit different KV-cache and attention paths; a model that's fast on `niah` can be slow on `code` at the same ctx size. Sweeping prompts is the difference between knowing how fast your model is and knowing how fast it is *for your workload*.

---

## Optional: concurrency ramp

If you want to know how the server holds up under multiple users, set `max_concurrency: N > 1` in the sweep block. Kata adds a rolling concurrency ramp (c=1 upward) per cell:

- Arrivals fire on an open-loop metronome at target rate `c / T_req(c=1)`
- In-flight is hard-capped at `c` — the level label is honest
- Per-c level, a hyperbolic `1 / (T_fixed + T_per_stream · c)` fit is verified against the next level's measurement; matches exit early to save wall time
- The ramp terminates on aggregate decline, plateau, per-request floor, or `max_concurrency`

This produces an operator verdict per cell — *concurrency wins* / *marginal* / *sequential wins* — based on whether the modeled ceiling beats the c=1 baseline. Useful if you're hosting multiple users on one box (small-team deploys, multi-tenant agents), and the honest answer is often "queue, don't parallelize" once context gets long.

Most single-user setups should leave `max_concurrency: 1`. The single-stream sweep is the load-bearing measurement.

---

## Configuration

Configs are YAML. A minimal one:

```yaml
name: "my benchmark"

hardware:
  cuda_visible_devices: "0"
  rig_label: "my-workstation"

server:
  launch: true
  endpoint: http://localhost:8080
  args:
    - --hf-repo: unsloth/Qwen3.6-27B-MTP-GGUF:IQ4_NL
    - --jinja
    - --batch-size: 8192
    - --ubatch-size: 2048
    - --fit: "off"                  # required — see "Fail loud" philosophy
    - -fa: "on"
    - --cache-type-k: q8_0
    - --cache-type-v: q8_0

sweep:
  context_sizes:  [2048, 8192, 16384]
  gen_sizes:      [512]
  prompt_presets: [niah, code]
  max_concurrency: 1               # bump to N for concurrency ramp

output:
  dir: runs
  live: true
  stream_tokens: true
  gpu_monitor: true
```

Kata auto-injects `--parallel`, `--ctx-size`, and `--kv_unified` from the sweep math so the server's KV budget is correctly sized. Don't set these manually unless you know why you're overriding the math.

Example configs live in `configs/`. The `qwen36-IQ4_NL-*.yaml` family covers the same model across rigs — useful templates for adapting to your hardware.

---

## Hardware notes

**Single-GPU**: the main path. Kata's measurements are validated on single-GPU configurations.

**Multi-GPU `--split-mode`**: usable, with caveats.

- `--split-mode layer` (default) — model layers split across GPUs sequentially. Low inter-GPU bandwidth needs. Works well even without P2P.
- `--split-mode row` — tensor-parallel row split. Heavy reduce-scatter + all-gather per layer. On hardware without P2P or NCCL, expect significant slowdown vs single-card single-stream.
- `--split-mode tensor` (experimental) — more granular all-tensor parallelism. Surprisingly viable on no-P2P hardware in our testing; smarter communication pattern than `row`.

For *clean* multi-GPU numbers, build llama.cpp with `-DGGML_CUDA_NCCL=ON` and verify NCCL is linked (`ldd llama-server | grep nccl`). Without it, llama.cpp falls back to a generic AllReduce that's significantly slower. Kata's startup VRAM check surfaces the NCCL state from the server's own log.

**RPC** (multi-machine via `rpc-server`): kata works with `--rpc <host>:<port>` configs. 1 Gb wired is fine for typical inference layer-split traffic; 10 Gb is the sweet spot; WiFi is unusable due to jitter. (Inference only — training has completely different bandwidth needs.)

---

## Output artifacts

| file | content | format |
|---|---|---|
| `raw.jsonl` | every completed request (timings, tokens, accept rate) | JSONL |
| `events.log` | every kata decision (cell starts, terminations, fit results, warnings) | JSONL |
| `summary.json` | machine-readable headline | JSON |
| `summary.md` | human-readable summary + per-cell tables | Markdown |
| `system.json` | rig manifest (CPU, RAM, GPUs, llama.cpp commit, driver, CUDA) | JSON |
| `server.log` | full llama-server stdout/stderr | text |
| `gpu_telemetry.csv` | per-GPU util/power/vram/temp sampled during the run | CSV |
| `plots/*.png` | throughput, prefill, TTFT, GPU timeline, fit residual | PNG |
| `narrative.md` | generated on demand (weatherman `n` key) | Markdown |

All artifacts are designed to be *posted* — git-committable, embeddable, shareable. The `raw.jsonl` + `events.log` + `system.json` triple is enough to reproduce any analysis kata does, and to scrutinize any number kata claims.

---

## What kata does *not* do (yet)

- **Training benchmarks.** Inference-only. Training has fundamentally different scaling characteristics; the abstractions wouldn't map.
- **Direct vLLM / SGLang / TGI support.** Kata measures `llama-server`. The methodology would translate (sweep matrix, fit model, artifact format are all stack-agnostic), but the adapter is unwritten. PRs welcome.
- **Multi-machine RPC stress benchmarks.** RPC works, but kata doesn't currently characterize the network-overhead axis as its own dimension.
- **Power-efficiency metrics (tokens per watt).** GPU telemetry is captured; the analysis is left to the operator. Could be a future plot type.
- **Continuous integration mode.** Kata is a benchmark harness, not a regression-test harness.

---

## Contributing

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for how to propose changes, what lands easily, and the methodology bar for new measurements.

Short version: open an issue first for anything bigger than a typo.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Acknowledgments

Built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) — kata exists because llama-server's diagnostic surface is already rich enough to support a real measurement harness. Thanks to the llama.cpp maintainers for the structured timings block, slot-level state in `/slots`, and the SSE completion stream that makes client-side TTFT measurement clean.
