# Contributing to katabasis

Thanks for the interest. Kata is opinionated — there are clear correctness, honesty, and operator-usability principles that govern what lands. Reading this first makes everything easier.

---

## TL;DR

- **Bug reports**: always welcome. Include the run dir (or `raw.jsonl` + `events.log` + `system.json`) so we can reproduce.
- **Bug fixes**: open a PR. Small ones with a clear root-cause analysis merge fastest.
- **New measurements / new plots**: open an issue first. Methodology bar applies (see below).
- **New backends (vLLM, SGLang, etc.)**: very welcome. Open an issue to discuss the adapter shape before coding.
- **Refactors**: rarely accepted on their own merit. Tie them to a bug or feature.
- **Typos / docs**: just send the PR.

---

## How to propose a change

1. **Open an issue describing the change and its motivation.** For anything bigger than a typo, the issue is the cheap-cycle review — better to align on approach before you write the code.

2. **Include real data, not just hypotheticals.** If you're proposing a new metric, a different aggregation, or a tuned threshold, attach a run dir (or the relevant artifacts from one) that motivates the change. *"This new termination criterion fires X% earlier on Y workload"* is a stronger argument than *"this would be more elegant."*

3. **Match the codebase's correctness/honesty bar.** Each design decision in kata is documented inline with the rationale (search for `audit` or `# Why:` in the code). New measurements should hold up to the same scrutiny:
   - Token-weighted aggregation for rates, never arithmetic mean of per-request rates
   - Honest distinction between measured and interpolated values
   - Failure modes surface as warnings, not silent degradation
   - Anything that's a heuristic should say so in a comment with the reasoning

4. **Keep PRs focused.** One commit per logical change. Squash if needed before merge. A 10-file PR doing five unrelated things is much harder to review than five focused PRs.

5. **Don't add dependencies casually.** Kata's deps are small and intentional (`pyyaml`, `requests`, `rich`, `matplotlib`, `numpy`, `textual`). Adding a new one needs justification.

---

## What lands easily

- Bug fixes with root-cause analysis (not just "this fixes it" but *why* the bug existed)
- New `--split-mode` characterization or hardware-specific notes that other operators will benefit from
- Better plot rendering, sparkline display, or weatherman UX polish
- New backend adapters (vLLM, SGLang, TGI) that produce the same analysis output (`summary.md`, `events.log`, plots) for stack-comparison purposes
- New config templates for popular models / hardware combinations
- Documentation improvements (especially "how to interpret X")

## What needs more discussion

- New termination criteria for the ramp scheduler (these compound; we want to make sure they generalize)
- Changes to the hyperbolic fit model, residual computation, or regime classifier (these affect every plot and the operator verdict)
- New summary.md sections (the artifact gets posted; layout changes need agreement)
- Refactors of the orchestrator (`kata.py:main`) — it's threaded carefully, easy to introduce races

## What doesn't land

- "Make the dashboard look prettier" without an operator-clarity argument (the audience is the operator, not the camera)
- "Use ${trendy library}" refactors with no functional improvement
- Adding test infrastructure without a discussion first (kata's test story is "real runs against real models" — synthetic unit tests for benchmark harnesses have value but the threshold is high)
- Removing the `--fit off` enforcement (this is policy — see the philosophy section in README)

---

## Reporting bugs

A useful bug report includes:

1. **What you ran**: command line, config used
2. **What happened**: the run dir, the failure mode, the warning message
3. **What you expected**: even one sentence helps
4. **System info**: copy from `system.json` if a run completed, or `nvidia-smi` + `python --version` + llama.cpp commit hash if it didn't

`raw.jsonl` rows + `events.log` rows + `system.json` are reproducibility gold. They're git-committable and embeddable in issues; please include them when relevant.

---

## Methodology bar for new measurements

Any new metric or termination criterion needs to:

1. **Be defensible against expert reply-questioning.** Imagine a contributor posting your new metric on r/LocalLlama with a 2-3 line explanation; would knowledgeable readers buy it, or would they spot a hole?

2. **Distinguish measured from derived/interpolated values.** If a value is computed from a fit or an interpolation, the operator needs to know it's not a direct measurement. `●` vs `◇` markers, `(interp)` suffixes, or similar.

3. **Have a known failure mode and a graceful fallback.** What happens with 2 samples instead of 30? With zero steady samples? With cells where the fit doesn't converge? Don't ship a metric that crashes or produces nonsense on edge cases.

4. **Be explainable in one paragraph.** If you need three paragraphs to explain why a number means what it means, the number is too complex for operator use. Either simplify the math or split it into multiple narrower metrics.

---

## Code style

- Format with whatever your editor does by default. Don't aggressively reformat existing files in PRs — those churn diffs hide the real change.
- Inline comments should explain *why*, not *what*. The code already says what.
- Type hints are encouraged but not required. Existing code is partially typed; match the surrounding style.
- Imports inside functions are OK when they're only used by that function (keeps top-of-file clean for occasional dependencies).
- `print()` is fine for CLI output. `console.print` from Rich for anything that needs styling.

---

## Project structure

```
kata/
├── kata.py          # main orchestrator: server launch, ramp scheduler, dashboard
├── bench_data.py    # data loading + aggregation helpers, fit + classifier
├── plot.py          # post-run PNG generation + summary.md writer
├── weatherman.py    # Textual TUI for reviewing runs
├── compare.py       # legacy A/B comparison (largely superseded by weatherman A/B)
├── sysinfo.py       # system manifest capture
├── prompts/         # prompt corpus builders (niah, code, chat)
├── configs/         # per-rig benchmark configs
└── runs/            # benchmark output (gitignored)
```

The main mental model: `kata.py` produces `runs/<ts>/raw.jsonl` and `events.log`; `plot.py` consumes those to produce `summary.md` and PNGs; `weatherman.py` is the interactive reviewer. All three share `bench_data.py` for the aggregation primitives.

---

## License

By contributing, you agree your changes are licensed under the project's MIT license. See [LICENSE](LICENSE).
