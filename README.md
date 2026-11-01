# Laya Ultrafast

**A local, open-weight browser agent using [Laya](https://huggingface.co/convaiinnovations/laya) as its decision model.**

Give it one goal. Laya, running in your process, picks an operation and an observed element. A separate small LLM
writes text only when the operation is `TYPE_TEXT`. No account or API key is needed for the decision model.

> [!IMPORTANT]
> **Status: the Laya integration is built and tested offline; a live Google Flights success with Laya has not been
> demonstrated yet.** See [Evidence and limits](#evidence-and-limits) for exactly what has and has not been measured.

## What this is

The browser loop is derived from Browser Use's [jev-ultrafast](https://github.com/browser-use/jev-ultrafast)
(MIT, © Browser Use), with the decision model replaced by a local one. The loop is inherited, not rewritten:

| Inherited from the Browser Use project | New in this project |
| --- | --- |
| One atomic DOM snapshot → indexed element table ([snapshot.js](laya_ultrafast/snapshot.js)) | In-process Laya decision model ([laya_backend.py](laya_ultrafast/laya_backend.py)) |
| Operation-specific target heads, one observation, one decision cycle | `DecisionModel` interface; every answer validated before an action can exist |
| Stale-page detection, freshness checks, click-occlusion checks, bounded execution | Coarse-to-fine selection for wide action spaces |
| Separate text helper, independent final verification, `DONE`/`BLOCKED` | Load/warm-up/decision/browser/text latency reported separately |
| Inspector, recording scripts, offline tests | `Task` abstraction; flights and a hotel task share one agent |

## What Laya does here

Laya is a non-autoregressive "System 1" model: one bidirectional-encoder forward pass answers several typed
questions with calibrated probabilities. It does not generate text. Each step it receives the observed state and
these questions, in a single call:

```text
page → element table → operation                 ┐
                       click_target              │ one Laya predict call
                       type_text_target          │ (one forward pass for pages that fit)
                       select_target, if present ┘
                                  │
                use only the selected operation's target
                                  │
              CLICK [7] ──────────┴──→ browser          TYPE_TEXT [3] ──→ small LLM → text → browser
```

The operations are `CLICK`, `TYPE_TEXT`, `SELECT`, `SCROLL_UP`, `SCROLL_DOWN`, `WAIT`, `DONE`, `BLOCKED`. Only
supported operations and targets are offered. Every element has one index; each target head contains only compatible
elements; native dropdown options carry an observed element/option index.

The model never emits selectors, coordinates, JavaScript, or shell commands. Its output is a choice among
offered indices, validated strictly ([details](docs/migration.md#response-validation)), and mapped back to an observed
DOM node. A malformed answer causes no browser mutation. There are no site-specific action scripts or prepared field
strings in the policy.

**Wide pages.** Laya's option budget is small, so a head wider than `max_options` is answered coarse-to-fine: chunks in
the same pass, then the chunk winners in one more. Narrow pages cost one forward pass.

## Why a separate text helper

Laya chooses; it does not write. When `TYPE_TEXT` is selected, a small OpenAI-compatible LLM receives the goal, the
field, visible page context, and recent actions, and must return exactly `{"text": "..."}` before anything is typed.
That keeps city names, dates, and search strings out of the decision layer, and lets the two models be swapped
independently. A generated value survives a stale-page retry only while the helper's entire input is identical.

## Install

Requires Python ≥ 3.12 and Chrome (or another Chromium) reachable by [Browser Harness](https://github.com/browser-use/browser-harness).

```bash
git clone <this repository>
cd laya-ultrafast
uv sync
cp .env.example .env     # set TEXT_MODEL_API_KEY for the text helper
uv run laya-ultrafast    # inspector at http://127.0.0.1:8766
```

`uv sync` installs `laya` and PyTorch. The default PyTorch wheel on Windows and macOS runs on CPU (Apple Silicon uses
PyTorch's MPS backend); for NVIDIA/CUDA install a CUDA build of PyTorch per [pytorch.org](https://pytorch.org/get-started/locally/).
Laya 0.3.26 has no MLX backend. The first run downloads the checkpoint from Hugging Face (about 0.6–0.8 GB).

Chrome connects through Browser Harness; run `uv run browser-harness --doctor` if it needs connecting, and allow
remote debugging when prompted. To use a separate browser instead of your own profile, start Chromium with
`--remote-debugging-port=9333 --user-data-dir=<throwaway dir>` and set `BU_CDP_URL=http://127.0.0.1:9333`.

**Behind a TLS-intercepting proxy?** The text helper verifies certificates against your operating system's trust
store (via `truststore`), so a CA your OS trusts works. Hugging Face downloads may still fail with
`CERTIFICATE_VERIFY_FAILED`. Do not disable verification: point `SSL_CERT_FILE` at a bundle that includes your CA, or
download the checkpoint files yourself and set `LAYA_MODEL` to that directory. On Windows, if Python crashes with
`OPENSSL_Applink`, rebuild the environment on a python.org Python: `uv sync --python <path to python.exe>`.

## Choose a checkpoint

```env
LAYA_MODEL=cklxx/laya-browser          # browser fine-tune (default, pinned to the measured commit)
# LAYA_MODEL=convaiinnovations/laya    # generic Laya
LAYA_DEVICE=auto                       # auto | cpu | cuda | cuda:0 | mps | xpu
LAYA_PRELOAD=1                         # load + warm up before the task clock starts
# LAYA_REVISION=<commit sha>           # blank pins the default to the measured commit; set it to pin another
```

The two are **not equivalent**, and the project reports which one ran (the Agent, inspector, and benchmark output
all print `checkpoint · kind · device`):

| | Generic Laya | `cklxx/laya-browser` |
| --- | --- | --- |
| Source | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), Apache-2.0 | [cklxx/laya-browser](https://huggingface.co/cklxx/laya-browser), Apache-2.0, **third-party** fine-tune of Laya |
| Trained on browser tasks | No | Yes (Mind2Web, NNetNav, WebChain, synthetic forms), in this request format |
| Token budget (`max_len` / `head_max_len`) | 512 / 192 | 1024 / 768 |
| `max_options` default | 12 | 60 |

Generic Laya's small head budget truncates most of the decision rules in each question
([why](docs/migration.md#request-mapping)). Its authors report 0.362 zero-shot accuracy on their typed-decisions
benchmark against 0.766 after fine-tuning, so do not assume generic Laya drives a browser well. The fine-tune's own
model card reports about 20–26% on its hardest held-out multi-step suite and notes it often says `DONE` too early.
**Test the checkpoint you configure**; `scripts/bench_decisions.py` does this offline.

Laya and `laya-browser` are Apache-2.0. The inherited browser loop is MIT, © Browser Use (see [LICENSE](LICENSE)). The compact
request format is adapted from the `laya-browser` reference wrapper. Training data licences for the fine-tune are
listed on its model card.

## Running on CPU (no GPU)

This project is meant to run locally, and a CPU is enough to run it, but not to run it fast. Expect:

- **First start:** about a minute to load the checkpoint (plus a download of roughly 0.6 GB the first time). The inspector
  loads it once at startup; `LAYA_PRELOAD=1` does the same for scripts, before any task clock starts.
- **Each decision:** seconds to tens of seconds on a 2-core laptop CPU (measured: 6–40 s, noisy), against tens of
  milliseconds the model's card reports on a high-end GPU. A task takes many decisions, so runs take minutes.
- **A busy machine can starve the browser.** In one measured run, inference saturated every core and the browser
  daemon timed out. Close other heavy programs while it runs.
- **Quality is experimental.** The default checkpoint did not complete the bundled local task in the one real-loop run, and
  made no sensible first move on four recorded pages. Treat results as something to inspect, not rely on.

`LAYA_HEAD_MAX_LEN` and `LAYA_PAGE_TEXT_CHARS` shrink what the model reads. In a small exploratory test, halving the head
budget (768 to 384) cut decision time by roughly 10–40% with the same decisions, and a much smaller one (192) changed
decisions without a clear speed-up. Four pages cannot say whether that is better or worse, and the checkpoint was trained
at 768, so these are unvalidated knobs, not defaults.

## Run Google Flights

```bash
uv run --env-file .env python examples/flights.py --keep-open
```

It runs the search, takes a **fresh** observation, independently checks the one-way setting, Zürich, London, September
20 2026, and visible matching flights, and saves a trace plus the metrics. It does not select or book a flight, and a
`DONE` choice alone never counts as success.

```python
from laya_ultrafast import Agent

with Agent("https://www.google.com/travel/flights?hl=en",
           "Find one-way flights from Zurich to London on September 20, 2026, for one adult in economy. "
           "Stop when matching flight options are visible.") as agent:
    for state in agent.run():
        print(state["elapsed_ms"], state["status"])
    print(agent.snapshot()["metrics"])
```

## Add a task

A task is data: a URL, a goal, and an independent check of the final page. There is one agent.

```python
from laya_ultrafast import Task, run_task

def verify(page):
    checks = {"on_article": page["url"].endswith("/Kurt_G%C3%B6del")}
    return {"passed": all(checks.values()), "checks": checks}

task = Task("wikipedia", "https://en.wikipedia.org/wiki/Main_Page",
            "Find and open the Wikipedia article about Kurt Gödel.", verify)
result = run_task(task)           # result["passed"] is the verifier's verdict, not the model's DONE
```

Never put site-specific steps or field values in a task; requirements are goal text.

## Hotels

Hotels are the next task, not a second agent. [examples/hotels.py](examples/hotels.py) runs the same `Agent` against the
bundled local stays page with a strict check, and builds goals for real sites:

```bash
uv run --env-file .env python examples/hotels.py --fixture
uv run --env-file .env python examples/hotels.py --url https://hotels.example --city Stockholm \
  --dates "14-18 November" --budget "under 800 GBP total" --rating "above 8" \
  --require "private bathroom" --require "free cancellation"
```

Search → filters → dates → guests → sort → inspection → opening a property uses the operations that already exist;
nothing in the Laya layer changes. Goals forbid booking, sign-in and payment, and the verifier fails a checkout page,
but **nothing in the executor technically prevents a model from clicking a booking button**. Real-site filter
correctness cannot be verified generically.

## Benchmarks

```bash
uv run python scripts/bench_decisions.py --repeats 20          # model only, offline, real recorded pages
uv run --env-file .env python scripts/measure_flights.py --output artifacts/flights/run-1   # one live, timed run
```

`measure_flights.py` records task time, decisions, model calls, browser protocol calls, text-helper calls, model
load and warm-up, and the independent verifier's verdict on a fresh observation. Results and caveats:
[performance-laya.md](docs/performance-laya.md).

## Small enough to read

| File | Job |
| --- | --- |
| [agent.py](laya_ultrafast/agent.py) | The complete loop, text-helper handoff, metrics |
| [model.py](laya_ultrafast/model.py) | `DecisionModel` interface, action space, request, strict validation |
| [laya_backend.py](laya_ultrafast/laya_backend.py) | In-process Laya: load once, compact request, coarse-to-fine |
| [text_helper.py](laya_ultrafast/text_helper.py) | The separate text LLM |
| [questions.py](laya_ultrafast/questions.py) | Decision rules and text instructions |
| [snapshot.js](laya_ultrafast/snapshot.js) · [browser.py](laya_ultrafast/browser.py) | Atomic DOM snapshot; execution and freshness |
| [task.py](laya_ultrafast/task.py) | `Task` and `run_task` with independent verification |
| [demo.py](laya_ultrafast/demo.py) | Local inspector |

Architecture and migration: [docs/migration.md](docs/migration.md) · [docs/design.md](docs/design.md).

## Evidence and limits

**Established** (details and caveats in [performance-laya.md](docs/performance-laya.md)):

- The integration works end to end: Laya loads once and is reused; operation and target decisions flow through strict
  validation to the executor; the text helper is invoked only for `TYPE_TEXT`; decision, browser and text-helper costs are
  reported separately. 78 offline tests pass, including 15 malformed-answer mutations that must never reach the browser.
- A real-loop run on the bundled stays page in headless Edge exercised all of this with the `cklxx/laya-browser` checkpoint.

**Not established, and the honest headline is negative so far:**

- **Decision quality.** On four recorded real pages, neither checkpoint chose a sensible first move (generic Laya
  answered `DONE`/`BLOCKED` everywhere; the fine-tune's moves were wrong too, identical to its author's own wrapper). In
  the real loop on the local stays page, the fine-tune did **not** complete the task: it oscillated between two links and
  never applied the requested category filter.
- **Latency.** On an i5-7200U laptop CPU a decision took about 6–40 s (noisy). That is the opposite of "ultrafast";
  GPU performance was not measured. The fine-tune's model card reports 22–27 ms on a high-end GPU.
- **Google Flights with Laya has not been run**: it needs a text-helper API key that was unavailable.

Treat Laya as a working *mechanism* whose decision quality you must measure on your own checkpoint and hardware before
trusting it.

The DOM reader handles common HTML and ARIA controls, not the full accessible-name specification. Shadow roots, frames,
canvas, uploads, pop-up tabs, nested scrolling, and arbitrary keyboard widgets are outside this MVP. Owned tabs share
the existing browser profile unless you point `BU_CDP_URL` elsewhere. A `DONE` choice still requires independent
outcome verification.

## Development

```bash
uv run ruff check .
uv run pytest
node --check laya_ultrafast/static/app.js
node --check laya_ultrafast/snapshot.js
uv build
```

Tests are offline (a fixture blocks sockets) and need no model weights or API keys. Live examples, recording scripts and
`scripts/measure_flights.py` call a browser and a paid text-helper API. Credentials and raw traces stay ignored.

---

[Browser Use](https://github.com/browser-use/browser-use) · [jev-ultrafast (original)](https://github.com/browser-use/jev-ultrafast) ·
[Browser Harness](https://github.com/browser-use/browser-harness) · [Laya](https://github.com/NandhaKishorM/laya) ·
[laya-browser](https://huggingface.co/cklxx/laya-browser)
