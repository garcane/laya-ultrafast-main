# Laya measurements

Everything here was measured on one machine and is a **small, noisy, CPU-only sample**. It supports no claim that Laya
is fast, accurate, or reliable on any task, and there is no comparison with another decision model.

## Setup

| | |
| --- | --- |
| Machine | Windows 10, Intel Core i5-7200U (2 cores / 4 threads, 2017 laptop), no GPU |
| Software | laya 0.3.26, torch 2.14.1+cpu (2 intra-op threads by default), Python 3.13.5 |
| Browser | Microsoft Edge 154, headless, throwaway profile, driven through Browser Harness (`BU_CDP_URL`) |
| Generic checkpoint | `convaiinnovations/laya` @ `7b928d828b7b0e022f929d9bd2e44165aa270148`, `model.safetensors` sha256 `891102d3…8d86` |
| Browser checkpoint | `cklxx/laya-browser` (v19s) @ `645cf366a2ae35f1086e8c20eff48f909bb49206`, `model.safetensors` sha256 `82844f25…8e5` |

Checkpoints were downloaded at those commits and loaded from local directories (this network's TLS chain is not in
the `certifi` bundle that the Hub client uses, so the Hub client could not download them).

## 1. Model only: recorded real pages (`scripts/bench_decisions.py`)

Four real observations recorded with `Browser.observe`, in [benchmarks/pages/](../benchmarks/pages): Google's cookie-consent
page before and after scrolling, the Google Flights home page (fresh profile; Google pre-filled the origin from this
network's location), and the bundled local stays page. A fifth, **synthetic** row adds 150 generic links to the
Flights page to exercise coarse-to-fine selection. `plausible` compares the first move with a short list I judged by
hand per page (stored beside each page). It is a smoke check on four states, not accuracy.

Load and warm-up, outside any decision:

| Checkpoint | Load | Warm-up |
| --- | ---: | ---: |
| Browser fine-tune | 58.6 s | 1.6 s |
| Generic | 24.6 s | 4.2 s |

Decisions (browser fine-tune, 10 repeats each; generic, 5 repeats each; `model_calls` is forward passes):

| Page | Elements | Browser fine-tune: first move | Plausible | Calls | Steady median |
| --- | ---: | --- | :-: | ---: | ---: |
| consent, buttons below the fold | 3 | CLICK "Language: English" | no | 1 | 6.3 s |
| consent, buttons visible | 5 | SCROLL_UP | no | 1 | 5.8 s |
| Flights home | 25 | SCROLL_DOWN | no | 1 | 11.4 s |
| local stays page | 10 | CLICK "View Casa Flora" | no | 1 | 11.8 s |
| Flights home + 150 synthetic links (synthetic) | 175 | SCROLL_DOWN | n/a | 2 | 27.5 s |

| Page | Generic: first move | Plausible | Calls | Steady median |
| --- | --- | :-: | ---: | ---: |
| consent, buttons below the fold | DONE | no | 1 | 10.2 s |
| consent, buttons visible | DONE | no | 1 | 9.8 s |
| Flights home | BLOCKED | no | 2 | 39.3 s |
| local stays page | DONE | no | 1 | 19.1 s |
| Flights home + 150 synthetic links (synthetic) | BLOCKED | n/a | 2 | 130.9 s |

What this shows, and does not:

- **Decisions were not usable in these four states.** Neither checkpoint chose a sensible first move on any page. Generic
  Laya, zero-shot, answered `DONE`/`BLOCKED` everywhere. The fine-tune's moves were also wrong here.
- **This is not an integration bug.** I ran the fine-tune author's own `decide()` (`laya_browser.py`) on the same four
  pages with the same weights. Its operation and target matched mine on all four.
- **CPU latency is nowhere near "ultrafast".** Seconds to tens of seconds per decision on this CPU, against 22–27 ms the
  fine-tune's card reports on a high-end GPU. Timing is noisy: the Flights page took about 11 s in one run and 28–44 s in a
  later one; I did not find a thread-count setting that helped (2, 4 and 1 threads were inconclusive). GPU numbers were
  not measured.
- Wide pages cost a second forward pass and roughly double or more the time. Generic Laya's smaller `max_options` (12)
  made even the 25-element Flights page need two passes.
- Warm-up removed no first-decision penalty at this scale: first and steady-state times were similar.
- Laya's token budget truncated the state once: generic Laya on the first consent page (512-token limit). The fine-tune
  truncated none of its benchmark decisions or the 38 decisions of the real-loop run.

## 2. Real loop: local stays page (fixture), browser fine-tune

The real `Agent` loop, real Laya, headless Edge, independent verifier. **Only the text helper was a local stand-in**
(a function returning `Lisbon` for the Destination field; it is not part of the product and I had no text-model key).
Laya made every operation and target choice.

- **Did not complete the task.** 38 decisions in 814 s. It chose `TYPE_TEXT` on Destination twice (stand-in helper called
  twice) and clicked Free cancellation, then oscillated between "View Casa Flora" and "Find a stay" and never selected the
  Design filter.
- The run ended when the browser daemon timed out (`Runtime.evaluate … 5s`) while model inference was saturating all four
  cores. The verifier could not read the final page, so the result is **not verified**; it is a failure either way, since
  the agent never reached `DONE` on a satisfied page.
- Metrics from this run: 38 model calls, median steady decision 19.6 s, 269 browser protocol calls totalling 33 s, 2
  text-helper calls, 0 truncated decisions. Decision time dominated everything.

## 3. CPU speed knobs (exploratory)

Browser fine-tune, same four pages, 3 repeats each, `--wide 0`. Smaller head budgets than the trained 768:

| `LAYA_HEAD_MAX_LEN` | Steady median per page | First moves judged sensible |
| ---: | --- | :-: |
| 768 (default) | 5.8 / 6.3 / 11.4 / 11.8 s | 0 of 4 |
| 384 | 5.5 / 5.9 / 6.7 / 9.6 s | 0 of 4 |
| 192 | 6.6 / 17.5 / 5.9 / 18.7 s | 2 of 4 |

Page order: consent (buttons visible), consent (buttons below the fold), Flights home, local stays page. The 192 run is
unreliable: the laptop appears to have suspended during it (it reported a model load of about 11 hours), and its timings
are erratic. Two sensible moves out of four pages is as consistent with luck as with a real effect, and the checkpoint was
trained at 768. Conclusion: no validated CPU speed-up; these remain optional, unvalidated settings.

## 4. Flights: not run

A live Google Flights run with Laya was **not** performed: it needs a text-helper API key (`TEXT_MODEL_API_KEY`), which
was unavailable. In a fresh profile, Google also first shows a cookie-consent page the agent must get past. No flight
task time, decision count, or reliability result exists for this project yet. `scripts/measure_flights.py` is the timed
live run; it has not been run against a live browser here.

## Reproduce

```bash
LAYA_MODEL=cklxx/laya-browser uv run python scripts/bench_decisions.py --repeats 10
LAYA_MODEL=convaiinnovations/laya uv run python scripts/bench_decisions.py --repeats 5
uv run --env-file .env python scripts/measure_flights.py --output artifacts/flights/run-1
```

Treat a handful of repeats on one page as an illustration. Anything stronger needs many more repeats, more tasks, and
better hardware than a CPU-only laptop.
