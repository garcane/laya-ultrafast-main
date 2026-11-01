# Replacing Jev with Laya

```text
Original:  Browser → TypeSafe Jev (HTTPS, API key) → operation + target → executor
New:       Browser → Laya (in this process)        → operation + target → executor
```

Only the decision layer changed. The observation (`snapshot.js`), the indexed action space, the freshness and
occlusion checks, the executor, the text helper, the independent verification, and the offline test approach are
inherited unchanged from [jev-ultrafast](https://github.com/browser-use/jev-ultrafast).

## The integration boundary

One method, defined in [model.py](../laya_ultrafast/model.py):

```python
class DecisionModel(Protocol):
    def choose(self, state: dict, questions: dict) -> dict: ...
```

`questions` is the same typed structure Jev received: an `operation` question, plus one `<operation>_target`
question per supported operation. The return value is `{"answers": {...}, "model": ..., "usage": ..., "info": ...}`
where each answer is `{"choice", "probabilities", "confidence"}`. That is both Laya's native `predict(state,
questions)` result and Jev's `/v1/systemone` wire shape, which is why almost nothing downstream moved.

`model.choose(page, goal, history, decider)` builds the request, calls the decider, and **validates every answer**.
The decider is never trusted: a backend bug cannot widen what the browser is allowed to do.
[laya_backend.py](../laya_ultrafast/laya_backend.py) is the only file that imports Laya.

## Model loading

`LayaDecisionModel` wraps `laya.load(checkpoint, device=..., revision=...)`.

- **Once per process.** `decision_model_from_env()` returns one shared instance per configuration. Every `Agent`
  and every decision reuses the loaded weights. Nothing is reloaded between steps.
- **Before the task clock.** With `LAYA_PRELOAD=1` (default) the model loads and runs one warm-up pass when the
  `Agent` is constructed, before a browser tab opens. With `LAYA_PRELOAD=0` it loads on the first decision; that
  load time is reported as `load_ms` and subtracted from that decision's latency.
- **Imported lazily.** `import laya_ultrafast` does not import torch (there is a test for this).
- **Pinning.** `LAYA_REVISION` pins a Hub commit. Local checkpoint directories are accepted.
- **Generic vs browser-fine-tuned** is read from the checkpoint's own `rl_agent_config.json` (`laya_fmt` is present
  only on browser fine-tunes), not guessed from its name. The kind appears in every decision, the inspector, and
  benchmark output.

## Request mapping

| Neutral request (`model.build_request`) | What Laya receives |
| --- | --- |
| `state.elements` (indexed table, up to 250) | A one-line `fields` summary of every form field and its value, placed first |
| `state.page.text` (≤6,000 visible chars) | First 1,200 chars |
| `state.recent_actions` | Last 10 |
| Target criteria `{"element", "role", "current_value", …}` | One compact string: `label (role) = 'value' checked=true` |
| Dropdown option `[2:3] Sort → Price` | `Sort → Price` |
| Rules (`questions.NEXT_ACTION`, `TARGET`) | Unchanged, in each question's instructions |

This compact form is the format `cklxx/laya-browser` was fine-tuned on. It is a lossless-enough compression of the
same observation, so the generic checkpoint receives the same request.

**Laya truncates instructions to its head budget.** Laya places the goal first and cuts the *end* of a question's
instructions when options need room. The rules are kept in the request, but a small head budget (generic Laya
defaults to 192 tokens) drops most of them. The browser fine-tune uses 768. This is a property of Laya, not
something this project can fix; it is the main reason generic Laya should not be assumed equivalent to a fine-tune.

## Large action spaces

A snapshot can expose up to 250 controls. Laya's option budget is far smaller, so a head wider than
`max_options` (60 for browser fine-tunes, 12 for generic Laya; `LAYA_MAX_OPTIONS` overrides) is answered
coarse-to-fine:

1. Split its options into interleaved chunks of at most `max_options`. All chunks are answered in the **same**
   forward pass as the operation question.
2. Take each chunk's winner and run **one more pass** where only the winners compete.
3. `p(option) = p(winner of its chunk) × p(option within its chunk)`, renormalised.

Narrow pages use exactly one forward pass. Wide pages use two; `model_calls` records which. The composed
probabilities are an approximation of a single wide decision, not a calibrated one. The winning index still maps
back to the exact observed element: chunking only changes which options are shown together.

## Response validation

`model.validate_choice` runs on the `operation` answer and on the target answer for the **selected** operation
only. An action can exist only when all hold:

- the answer is a mapping with `choice`, `probabilities`, `confidence`;
- `choice` is in the currently offered set (so an unknown operation or target is rejected);
- the probabilities cover **exactly** the offered choices;
- every probability and the confidence is a finite `int`/`float` in `[0, 1]` (booleans rejected);
- the probabilities sum to 1 (±0.02);
- the selected choice is a highest-probability choice;
- the head that the selected operation consumes did not **collapse**: if Laya's head budget left two options with
  identical token spans, the model cannot tell them apart, so the decision is refused (a collapsed head the
  operation does not consume is ignored).

The target index then maps through the observed action table to an action id, and from there to an observed DOM
node. The executor re-checks page freshness, visibility, and click occlusion immediately before input.

## Failure handling

| Failure | Outcome |
| --- | --- |
| Checkpoint missing / offline first download | `RuntimeError` naming the checkpoint; no tab, no action |
| Backend exception | Propagates; no action, decision cleared |
| Malformed or invalid answer | `ValueError`; no action, decision cleared |
| Collapsed consumed head | `ValueError`; no action |
| Page changed during inference | Existing stale-page path: re-observe, decide again, no action |
| Truncated state | Allowed, but recorded in `decision["info"]["truncated"]` |

Browser mutations are still never retried, and execution is still logged before the result is observed.

## Latency measurement

`Agent.snapshot()["metrics"]` keeps costs separate:

| Metric | Meaning |
| --- | --- |
| `model_load_ms`, `model_warmup_ms` | One-off, outside the task clock when preloaded |
| `decisions`, `model_calls` | Decisions made; forward passes used (1 or 2 each) |
| `first_decision_ms`, `steady_decision_median_ms` | First decision vs the rest |
| `decision_ms` | Total time inside `DecisionModel.choose` |
| `browser_calls`, `browser_ms` | Browser protocol calls and time inside them |
| `text_helper_calls`, `text_helper_ms` | Text-helper requests and time |

`scripts/bench_decisions.py` measures the model alone on recorded real pages. `scripts/compare_flights.py`
alternates frozen-Jev and Laya runs of the live flight task. Results: [performance-laya.md](performance-laya.md).

## Why in-process, and what was not kept

The goal is low-latency local inference, so the decision model runs in the agent's process. An HTTP hop would add a
round trip and a second process to manage, and buys nothing here. `laya-serve` exposes `/v1/systemone` and
`cklxx/laya-browser` ships a Jev-compatible server, so an HTTP `DecisionModel` is a thin class if someone needs a
shared GPU box; it is not included. The TypeSafe endpoint, `TYPESAFE_API_KEY`, and `TYPESAFE_MODEL` are removed
(a test asserts the package contains none of them). The original Jev implementation remains in git history
(`f2fbdd5`) for the benchmark.
