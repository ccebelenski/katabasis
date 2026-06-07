# llama.cpp Benchmark Suite ("katabasis") — Plan

## Context

The repo today is just launch scripts (`gptoss120.sh`, `lfm2-24b.sh`, etc.) that wrap `llama-server` with different flags. You want a repeatable, video-friendly benchmark harness that:

- Sweeps **prompt size × generation size** for prefill and decode throughput (what `llama-bench` does)
- Supports **MTP / draft (speculative) models** (which `llama-bench` currently can't)
- Is driven by a **simple YAML config** so a "run" is reproducible and editable on camera
- **Accumulates rounds** and renders graphs across them
- Looks compelling on video: live terminal dashboard + final graphs, while still exposing the actual HTTP calls and raw timings so a technical viewer can follow what's happening

The differentiator (vs other AI YouTubers and vs `llama-bench`):
- Live **speedometer panel** for prefill / decode t/s during the sweep
- A **draft-acceptance visualization** — per-token accept/reject stream, accept rate per draft position, effective speedup vs no-draft baseline. Nobody is showing this yet because the tooling doesn't exist.
- Reproducible YAML configs you can hand viewers

`llama-ccbench` (the project you mentioned) — quick web check during Phase 1, but plan does not depend on it. If it already does most of this we vendor/extend; otherwise we build our own. Default assumption: build our own.

## Architecture

```
bench.yaml ──► kata.py ──► launches llama-server (optional)
                  │            │
                  │            ▼
                  │       /health wait + system manifest
                  │            │
                  ▼            ▼
          sweep loop ──► native /completion (per request)
                  │            │
                  ▼            ▼
          live rich UI    runs/<ts>/{raw.jsonl, system.json,
                                     gpu_telemetry.csv, server.log}
                                │
                                ▼
                           plot.py ──► runs/<ts>/plots/*.png + summary.md
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
        compare.py                          weatherman.py (TUI)
       (CLI A/B diff)                  ┌── indexes runs/* ──┐
                                       │ tree by model/quant│
                                       │ filter/sort/pin    │
                                       │ summary / charts / │
                                       │ raw  / A/B tabs    │
                                       └────────────────────┘
```

Data path per request:
```
prompt (sized) ─► POST /completion ─► response{timings, tokens, draft_*}
                                              │
                                              ├─► live panel update (rich)
                                              └─► append JSONL row
```

## Files to create

All Python, kept in repo root next to existing `.sh` scripts.

| File | Purpose |
|---|---|
| `bench.yaml` | Example config (committed) |
| `kata.py` | Orchestrator: loads YAML, optionally launches server, runs sweep, live dashboard, writes JSONL |
| `plot.py` | Reads `runs/<ts>/raw.jsonl`, emits PNGs (matplotlib) |
| `compare.py` | Diffs N run dirs, side-by-side bars / overlay lines |
| `weatherman.py` | Textual TUI for reviewing/comparing runs across models, quants, rigs — the on-camera payoff |
| `requirements.txt` | `pyyaml`, `requests`, `rich`, `matplotlib`, `numpy`, `textual`, `plotext` |
| `README.md` | One-page "how to record a video" walkthrough |

No new dependencies on llama.cpp source — talks to `llama-server` over HTTP only.

## `bench.yaml` schema

```yaml
name: "gpt-oss-120b vs draft-8b"

# Hardware selection — applied as env vars before launching the server,
# and recorded into the run manifest so cross-system comparisons are valid.
hardware:
  cuda_visible_devices: "0,1"   # sets CUDA_VISIBLE_DEVICES (and HIP_VISIBLE_DEVICES)
  # Optional, passed through to llama-server if set:
  tensor_split: "50,50"         # → --tensor-split
  main_gpu: 0                   # → --main-gpu
  # Free-form label that ends up in plot titles and filenames:
  rig_label: "dual-3090"

server:
  launch: true
  binary: /mnt/ssd/llama.cpp/llama-server
  args:
    - --hf-repo: unsloth/gpt-oss-120b-GGUF:F16
    - --ctx-size: 131072
    - --batch-size: 8192
    - --ubatch-size: 2048
    - -fa: auto
    # MTP / draft model knobs (the whole point)
    - --model-draft: /path/to/draft.gguf
    - --draft-max: 16
    - --draft-min: 4
    - --draft-p-min: 0.6
  endpoint: http://localhost:8080
  health_timeout_s: 120

sweep:
  prompt_sizes:  [128, 512, 2048, 8192, 32768]
  gen_sizes:     [64, 256, 1024]
  rounds:        3
  warmup_rounds: 1
  prompt_source: lorem        # or: file: path/to/seed.txt

output:
  dir: runs                   # runs/<timestamp>__<rig_label>/...
  live: true                  # show rich dashboard
  stream_tokens: true         # spool decoded tokens to a side panel
```

A second config `bench.draft-off.yaml` reuses the same sweep but with draft disabled — that's the A/B you record.

## `kata.py` — what it does, step by step

1. **Parse YAML** (`pyyaml`). Print the resolved config as a `rich` Panel so the viewer sees exactly what's about to run.
2. **Capture system manifest** (see "System manifest" section below). Write `runs/<ts>/system.json`. Render a one-screen `rich` Panel — CPU, RAM, OS/kernel, llama.cpp commit, and a per-GPU table — so the viewer sees the rig on camera before any numbers are produced.
3. **Server**: if `launch: true`, set `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` from `hardware.cuda_visible_devices` in the child env, append `--tensor-split` / `--main-gpu` to args if specified, then `subprocess.Popen` the binary with assembled args (echoed to console first — this is the technical-transparency bit). Poll `GET /health` until ready. Tee server stderr to `runs/<ts>/server.log`.
4. **Build prompts**: generate a corpus longer than max prompt size, then truncate by tokens. Use `/tokenize` endpoint on llama-server to get exact token counts (don't approximate — viewer can verify).
5. **Warmup**: run `warmup_rounds` worth of (smallest prompt, smallest gen) requests, discard results.
6. **Sweep**: nested loop over `rounds × prompt_sizes × gen_sizes`. For each:
   - POST to native `/completion` with `{"prompt": ..., "n_predict": gen, "cache_prompt": false, "stream": true}` (streaming so the token panel updates live; `cache_prompt: false` to actually measure prefill each time).
   - Aggregate streamed chunks. Final chunk contains `timings { prompt_n, prompt_per_second, predicted_n, predicted_per_second, ... }` and, when a draft model is loaded, `draft_n` / `draft_n_accepted` (confirm exact field names against current llama.cpp during impl — fall back to scraping `/slots` if absent).
   - Append one JSONL row: `{round, prompt_size, gen_size, prefill_tps, decode_tps, draft_n, draft_accepted, ts, raw_timings}`.
7. **Live dashboard** (rich `Live` + `Layout`):
   ```
   ┌─ config ───────────┬─ sweep progress ──────────┐
   │ model: gpt-oss…    │ ███████░░░ 18/30          │
   │ draft: yes         │ now: prompt=2048 gen=256  │
   ├─ live speedo ──────┼─ token stream ────────────┤
   │ prefill: 1820 t/s  │ "...the quick brown fox…" │
   │ decode:   142 t/s  │                           │
   │ accept:    68%     │                           │
   ├─ rolling chart (plotext ascii) ────────────────┤
   │  decode t/s over last 30 reqs                  │
   │  ▂▃▅▆▇▆▅▆▇▇▆▅▆▇                                │
   └────────────────────────────────────────────────┘
   ```
   Uses `rich` for layout + `plotext` for inline ascii charts. Looks great on video without leaving the terminal.
8. **Teardown**: kill server if we launched it. Print final summary table and the path to results.
9. **Auto-plot**: shell out to `plot.py runs/<ts>/`.

## System manifest (`runs/<ts>/system.json`)

Captured once at startup. Everything is best-effort — missing tools degrade to `null`, never crash. All shelled out via `subprocess` so a viewer reading the code can see exactly where each number came from.

Fields:

```json
{
  "rig_label": "dual-3090",
  "timestamp_utc": "...",
  "host": {
    "hostname": "...", "os": "Linux", "kernel": "6.18.5",
    "distro": "...", "python": "3.12.x"
  },
  "cpu": {
    "model": "...",            // /proc/cpuinfo "model name"
    "cores_physical": N, "cores_logical": M,
    "max_mhz": ...,            // lscpu
    "governor": "performance"  // cpupower / sysfs
  },
  "memory": { "total_gb": ..., "available_gb": ... },   // /proc/meminfo
  "selection": {
    "cuda_visible_devices": "0,1",
    "tensor_split": "50,50",
    "main_gpu": 0
  },
  "gpus": [                    // nvidia-smi --query-gpu=... --format=csv,noheader
    {
      "index": 0, "uuid": "...", "name": "NVIDIA GeForce RTX 3090",
      "vram_total_mib": 24576, "vram_free_mib": 23800,
      "driver": "550.xx", "cuda": "12.4",
      "pstate": "P0", "power_limit_w": 350,
      "sm_clock_mhz": 1695, "mem_clock_mhz": 9751,
      "pcie_gen": 4, "pcie_width": 16
    }
  ],
  "rocm_gpus": [...],          // rocm-smi if present, else []
  "llama_cpp": {
    "binary": "/mnt/ssd/llama.cpp/llama-server",
    "version_line": "...",     // llama-server --version
    "git_commit": "...",       // llama-server --version parses commit; else null
    "build_flags": "CUDA VULKAN FA_ALL_QUANTS"  // parsed from --version
  }
}
```

Capture sources (one helper per tool, all optional):
- `nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,driver_version,pstate,power.limit,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits`
- `nvidia-smi -q -d COMPUTE` for CUDA runtime line, or `nvcc --version`
- `rocm-smi --showproductname --showmeminfo vram --showdriverversion` (only if binary present)
- `lscpu`, `/proc/cpuinfo`, `/proc/meminfo`
- `uname -a`, `cat /etc/os-release`
- `<binary> --version` (llama-server prints version + commit + build flags)

GPU selection enforcement:
- `hardware.cuda_visible_devices` is set in the **child process env** for the server, not the katabasis process — so katabasis can still query all GPUs for the manifest while the server only sees the selected ones.
- The manifest records **both** the full GPU list and the `selection` subset so plots can title themselves "RTX 3090 #0 only" vs "dual 3090".
- If the user sets `tensor_split` / `main_gpu` in YAML, those are appended to llama-server args; we don't try to second-guess the user.

Live mini-monitor (optional, off by default, toggle via `output.gpu_monitor: true`):
- Background thread polls `nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.used --format=csv,noheader,nounits -l 1` and appends rows to `runs/<ts>/gpu_telemetry.csv`. Shown as a tiny sparkline in the dashboard. Plot.py renders it as an overlay on the decode-tps timeline.

## `plot.py` — graphs

Reads `raw.jsonl` and `system.json` from a run dir, produces in `runs/<ts>/plots/`. Every plot footer auto-stamps: `rig_label • GPU model(s) selected • llama.cpp commit • build flags` — so screenshots are self-documenting when they end up on Twitter.

1. **`prefill_vs_prompt.png`** — line, x=prompt_size (log), y=prefill t/s, error bars over rounds, one series per gen_size (or single series with gen_size collapsed).
2. **`decode_vs_gen.png`** — line, x=gen_size, y=decode t/s, one series per prompt_size.
3. **`heatmap_decode.png`** — 2D heatmap, axes (prompt_size, gen_size), color=decode t/s. Single image that summarizes a whole run.
4. **`draft_accept_rate.png`** (only if draft enabled) — bar per request showing acceptance %, plus mean line. Histogram of draft burst lengths.
5. **`gpu_timeline.png`** (only if `gpu_telemetry.csv` exists) — util %, power draw, VRAM used vs wall time, with sweep request boundaries marked.
6. **`summary.md`** — markdown table of means/std-devs **plus a "System" header block** lifted from `system.json` so the file is paste-ready into a video description or forum post.

Style: dark background, large fonts, high-contrast colors — readable when downscaled to 1080p in OBS.

## `compare.py` — A/B

```
python compare.py runs/2026-06-03_draft-on runs/2026-06-03_draft-off
```

Produces `comparison/`:
- Grouped bar: decode t/s per (prompt, gen) cell, two bars per cell (draft on/off, or rig-A vs rig-B)
- Single big number: **effective speedup** = decode_tps_a / decode_tps_b, averaged
- Overlay line plots
- A **system diff table** built from each run's `system.json` — highlights what changed (GPU, driver, commit, flags) so cross-rig comparisons are honest. Refuses to render a single "speedup" number when both rig labels differ AND configs differ, to avoid misleading apples-to-oranges plots; emits a warning panel instead.

This is the money shot of the video.

## `weatherman.py` — TUI for reviewing runs (the on-camera payoff)

A Textual app. Launched as `python weatherman.py [runs_dir]` (default `runs/`). Designed assuming a camera is pointed at it. Big fonts, clear keybindings shown at the bottom, no mouse required.

### Indexing

On startup, scans the runs dir and reads each `system.json` + `summary.json` (the latter written by `plot.py`) into an in-memory index. Each run has:

- `model` (parsed from server args: `--hf-repo` or `--model`)
- `quant` (parsed from filename: `Q4_K_M`, `Q8_0`, `F16`, ...)
- `draft_model` (if any)
- `rig_label`
- `gpus_selected` (joined name string, e.g. "2× RTX 3090")
- `decode_tps_mean`, `prefill_tps_mean`, `accept_rate_mean`
- `commit`, `timestamp`

A filesystem watcher (Textual's interval timer + mtime poll on the runs dir) picks up newly completed runs without restart — so you can kick off `kata.py` in one terminal and have the new entry pop into weatherman on camera.

### Layout

```
┌─ weatherman ───────────────────────────────────────────────────────────┐
│ [filter: gpt-oss____________]  models▾  quants▾  rigs▾  draft▾         │
├─ runs ──────────────────┬─ detail ──────────────────────────────────── │
│ ▾ gpt-oss-120b          │  ╭─ summary ─╮ ╭─ charts ─╮ ╭─ raw ─╮ ╭─ A/B╮│
│   ▾ F16                 │  rig: dual-3090                              │
│     ● 2026-06-03 dual…  │  gpu: 2× RTX 3090 (24G)                      │
│     ○ 2026-06-02 dual…  │  commit: a1b2c3d                              │
│   ▾ Q8_0                │  decode: 142 t/s  prefill: 1820 t/s          │
│     ○ 2026-06-01 a100   │  accept: 68%                                 │
│ ▾ Hermes-4.3-36B        │  ─────────────────────────────────────────── │
│   ▾ Q8_0                │  ▆▇▇▆▅▆▇▇▆▅  decode t/s over sweep            │
│     ○ 2026-05-30 dual…  │  ░░▒▒▓▓██▓▓  prefill heatmap                  │
│ ▾ LFM2-24B              │                                              │
│   …                     │                                              │
├─────────────────────────┴──────────────────────────────────────────────┤
│ ↑/↓ navigate  ⏎ open  space pin (A/B)  c compare  p PNG  / filter  q quit │
└────────────────────────────────────────────────────────────────────────┘
```

Three views in the right pane (tab keys 1/2/3/4):

1. **Summary** — big-text panel: model • quant • rig • headline numbers. The "weather forecast" card.
2. **Charts** — plotext renderings of the same data `plot.py` produced (prefill-vs-prompt, decode-vs-gen, heatmap, accept-rate). Plotext draws into the terminal so this works in any modern terminal without Sixel/Kitty graphics support. Pressing `p` opens the matplotlib PNG via `xdg-open` for a hi-res cut-in.
3. **Raw** — scrollable DataTable of the JSONL rows. For the technical viewer who wants to see numbers.
4. **A/B** — only enabled when 2+ runs are pinned. Overlay charts + system-diff table + headline speedup (reuses `compare.py` internals as a library).

### Cross-model comparison

The tree groups by model so "different models against each other" is the natural workflow: pin one run from gpt-oss-120b/F16 and one from Hermes-4.3-36B/Q8_0, press `c`, A/B view appears. The system-diff table from `compare.py` makes the apples-to-oranges nature explicit (different models, possibly different rigs).

Sort/filter bar at the top:
- Free-text filter (matches model, quant, rig)
- Dropdowns: model, quant, rig, draft on/off
- Sort: by date, by decode t/s, by accept rate

### Presenter mode (`F1` toggle)

Hides timestamps and file paths, enlarges fonts (Textual CSS swap), shows only model • quant • rig • headline numbers. For pulling up a single run as a "result card" on stream. Default off so the technical view is the landing experience.

### Reuse

- `compare.py` is refactored so its plotting + diff functions are importable; `weatherman.py` calls them. CLI behavior of `compare.py` is unchanged.
- `plot.py`'s data-loading helpers (jsonl → pandas/dict) get extracted into a small `bench_data.py` module that both `plot.py` and `weatherman.py` import. No duplication.

### Why this matters on camera

It's the deliverable side of the workflow. After a 20-minute benchmark, instead of cutting to matplotlib windows or a text dump, you switch terminals to weatherman and **navigate** the result space — pin two runs, hit `c`, talk through the diff. That's a 30-second payoff segment that nobody else has because nobody has the tool.

## Reuse / things to lean on

- `rich` — `Live`, `Layout`, `Panel`, `Table`, `Progress`. Standard.
- `textual` — TUI framework for `weatherman.py` (tree, tabs, datatable, filter input).
- `plotext` — ascii charts inside both the live `rich` dashboard and Textual panels. Works in any terminal.
- `matplotlib` — final PNGs (and the hi-res cut-in from weatherman via `xdg-open`).
- llama-server's **native `/completion` endpoint** (not OpenAI-compatible) — that's where prefill/decode timings live. The existing `x.sh` uses `/v1/chat/completions` which lacks per-stage timings; we deliberately switch.
- llama-server's **`/tokenize`** endpoint for exact prompt sizing.
- The existing launch scripts (`gptoss120.sh`, etc.) as reference for the flag sets that work on your hardware — we mirror them as YAML.

## Verification

End-to-end test for whoever implements this:

1. Build/install deps: `pip install -r requirements.txt`.
2. **Manifest-only dry run**: `python kata.py bench.yaml --manifest-only` writes `system.json` without running a sweep. Inspect manually: every GPU listed, driver/CUDA populated, llama.cpp commit non-null. This is what proves system-info capture works before you waste a benchmark run.
3. Manual smoke test against an already-running server:
   - Start a small model by hand: `./lfm2-24b.sh &`
   - Set `bench.yaml` with `launch: false`, `endpoint: http://localhost:8080`, tiny sweep `prompt_sizes: [128, 512]`, `gen_sizes: [64]`, `rounds: 1`.
   - Run `python kata.py bench.yaml`. Expect: live dashboard renders, completes ~4 requests, `runs/<ts>/raw.jsonl` has 4 rows with non-zero `prefill_tps` and `decode_tps`, plots appear in `runs/<ts>/plots/`, every plot footer shows rig label + GPU + commit.
4. Self-launch test: same config but `launch: true` with the lfm2 args. Confirm server starts, health passes, sweep runs, server is killed cleanly on exit and on Ctrl-C (signal handler).
5. **GPU selection test**: on a multi-GPU box, set `hardware.cuda_visible_devices: "1"` and confirm via `nvidia-smi` during the run that only GPU 1 has VRAM allocated. Confirm `system.json` records both the full GPU list and the selection.
6. Draft-model test: point `--model-draft` at a small draft model with a larger main model, run sweep, confirm `draft_accept_rate.png` is produced and acceptance % is in a plausible range (40–80%).
7. A/B: run twice (draft on, draft off), then `python compare.py runs/<a> runs/<b>` and confirm `comparison/` is populated, the speedup number is sane, and the system-diff table shows only the draft-model line as changed.
8. **Cross-rig test**: copy a `runs/` dir from a second machine into this one and run `compare.py` across them — confirm the system-diff table highlights GPU/driver differences and that a misleading single "speedup" number is suppressed.
9. **Weatherman smoke test**: `python weatherman.py runs/` after ≥3 runs (different models or quants) exist. Confirm: tree groups by model→quant, filter narrows the list, ⏎ opens Summary, 2 switches to Charts and plotext renders, space pins, `c` opens A/B with overlay charts and system-diff table. Then kick off a new `kata.py` run in another terminal and confirm the new entry appears in weatherman within ~10s without restarting it. Toggle `F1` presenter mode and verify the layout still looks good on camera.
10. Recording rehearsal: dry-run the whole flow in one shell session top-to-bottom (katabasis in pane 1, weatherman in pane 2). If anything looks ugly on a 1080p capture (cramped layout, dim colors, scrollback noise, slow Textual reflows), iterate before recording for real.

## Out of scope (don't do these now)

- Web dashboard — terminal-only is more authentic on video.
- Comparing against vLLM / TGI — separate project.
- Auto-tuning ubatch/batch — we expose them in YAML, viewer changes them between takes.
- Per-SM / per-kernel GPU profiling (nsys, ncu) — out of scope; this is throughput, not microarch.

## Post-plan changes (record of what shipped differently)

- **Schema rename.** `prompt_sizes` → `context_sizes`, `prompt_source` → `prompt_presets` (list). What the sweep varies is the *input context length at decode-start*, not "the prompt" — `context` is the honest term. Old keys still work via back-compat shims in `bench_data.get_context_sizes` / `get_prompt_presets`.
- **Prompt presets.** `prompt_presets` is a list of named corpora that the sweep iterates as an outer axis. Built-ins: `lorem` (no task), `code` (`prompts/code_corpus.py` + refactor request), `chat` (`prompts/chat_corpus.txt` + summarize request). The `file:` form passes through a user-supplied corpus. Each raw.jsonl row gets a `preset` field; `plot.py` facets PNGs per preset and emits overlay charts comparing presets on a single axis. The biggest motivation is honest speculative-decoding evaluation — output domain (code vs prose) drives draft acceptance by 2-3×, and a single-domain benchmark hides that.
- **Bundled corpora.** `prompts/code_corpus.py` is generated by concatenating the project's own `.py` files (~110 KB, ~32K tokens). `prompts/chat_corpus.txt` is a 6-conversation synthetic dialogue (~28 KB). Both recycle with a separator marker for context targets that exceed the corpus length; documented limit ~12K tokens for clean coverage.
- **Per-preset breakouts.** `weatherman.py`'s `SummaryPanel` shows a per-preset stats block when multiple presets are present in a run. `plot.py`'s `summary.md` gets a per-preset section. The end-of-run table in `kata.py` stays collapsed (one row per ctx/gen, averaging across presets) on user request — per-preset breakdown is in `plots/summary.md` and weatherman.
- **Visual changes (predate the preset work but worth recording).** The live dashboard moved from a plotext rolling chart to nvtop-style Braille sparklines for GPU (util/pwr/vram/temp) and bench metrics. The weatherman ChartsPanel and ABPanel use a custom Braille line-chart renderer in `bench_data.line_chart`. `plotext` dropped from `requirements.txt`.
- **`configs/` subdirectory.** Per-model / per-setup YAMLs live in `configs/<name>.yaml`. The root `bench.yaml` remains as a commented example.

## Post-plan changes — second round (2026-06-03)

- **Project renamed `benchy` → `ccbench`.** The name "benchy" was already taken by another OSS project. `ccbench` = user's initials. `benchy.py` is now `ccbench.py`. `weatherman` kept its name.
- **Project renamed `ccbench` → `katabasis` (2026-06-05).** Generic `ccbench` swapped for `katabasis` — the descent. Benchmarking is a journey into the underworld: hazards, traps, and the criticism inherent in measurement. `ccbench.py` is now `kata.py` (CLI: `python kata.py …`); short form `kata`. `weatherman` still kept its name.
- **`lorem` preset removed**, replaced by `niah` (needle-in-a-haystack). Lorem deterministically EOS'd at 3 tokens with the Qwen3.6-MTP model, polluting decode-rate aggregates with cold-start bias. NIAH uses a Pride & Prejudice excerpt (`prompts/niah_corpus.txt`) with an anchor passage at ~50% depth (opening of Chapter XXXV — Darcy's letter) and rotates through 3 retrieval questions. The anchor is preserved in every NIAH prompt regardless of target ctx, so the answer is always findable.
- **Token-weighted aggregation.** `aggregate_by_cell` now computes `sum(tokens) / sum(time)` rather than arithmetic mean of per-request rates. Arithmetic mean was inflated by short-EOS requests (cold-start tokens decode faster than steady-state). New `decode_ms_per_token` field surfaces the latency-domain rate.
- **TTFT metric.** Client-side wall time from POST to first non-empty content chunk, recorded per request as `ttft_ms`. Added to raw.jsonl rows, summary table column, summary.json headline, weatherman SummaryPanel, ChartsPanel, and a new `ttft_vs_context.png` chart.
- **Concurrency axis.** New `concurrency_levels: [1, 4, 8]` YAML key. ThreadPoolExecutor closed-loop: at concurrency N, N requests fire in parallel per batch. Each request becomes one raw.jsonl row tagged with `concurrency`, `batch_id`, `request_index`, `batch_wall_s`, `max_observed_concurrency`. New `throughput_vs_concurrency.png` and `ttft_vs_concurrency.png` plots.
- **Auto-control of llama-server flags.** `assemble_server_argv` now bumps `--parallel` to `max(concurrency_levels)`, auto-sizes `--ctx-size` to `(max_ctx + max_gen) × max_c × 1.10`, and injects `--kv_unified` when c > 1. Never reduces user-set values. Critical fix — undersized `--ctx-size` silently cancels server-side requests at high concurrency, dropping data.
- **Concurrency gating.** Between concurrency-level *decreases* (c=8 → c=1), katabasis polls `/slots` until idle (2 s timeout) before starting the next batch. Asymmetric (only on decrease) — c=1 → c=8 doesn't need gating because the new batch saturates anyway. Default `gate_timeout_s: 2.0`, `gate_only_on_decrease: true`. Diagnostic counters at run end track empty-timings rows and over-busy batches.
- **Dashboard honesty pass.** Bench panel sparkline deques clear on concurrency-level change, with the just-completed level's means snapshotted as a pinned "prior c=N" summary row at the top. New `agg t/s` line per batch shows the workgroup-throughput story rising with c while per-request decode drops. `ms/tok` inline on decode line.
- **Dashboard width-aware.** Sparkline width is `max(20, min(150, console.width − 59))`, recomputed on every render so terminal resizes are respected.
- **Local-time display.** `bench_data.format_timestamp_local` converts the UTC `timestamp_utc` in `system.json` to system local time for all UI display. Storage stays UTC for portability.
- **Tokens panel** title becomes `tokens (slot 0 of N)` at c > 1 so the viewer knows they're seeing 1 of N parallel streams.
- **`weatherman.py` rebuilt** for the new schema: per-preset block, per-concurrency block (`aggregate t/s`, `per-req t/s`, `TTFT mean/p95`), scrollable panels via VerticalScroll, PNG-open candidates list (multi-preset filenames), tree leaves get inline sparklines and `spec=...` hints.
- **Back-compat removed (2026-06-03 cleanup).** Old runs were wiped; legacy keys (`prompt_sizes`, `prompt_source`, `prompt_size` in raw rows) no longer accepted. Schema is solely `context_sizes` / `prompt_presets` / `context_size`. `LOREM_FALLBACK` removed.
