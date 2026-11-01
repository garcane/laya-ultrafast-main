"""Model-only benchmark: no browser, no text helper, no paid API. Replays recorded real pages through Laya.

    LAYA_MODEL=cklxx/laya-browser uv run python scripts/bench_decisions.py --repeats 20

Reports load and warm-up separately from decision latency, then the first decision against steady state.
`plausible` compares the first move with a short hand-judged list in benchmarks/pages/*.json. That is a smoke
check on four real states, not accuracy: no claim about task success follows from it.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.flights import GOALS as FLIGHTS_GOAL  # noqa: E402
from examples.hotels import FIXTURE_GOAL  # noqa: E402
from laya_ultrafast import model  # noqa: E402
from laya_ultrafast.laya_backend import decision_model_from_env  # noqa: E402

GOALS = {"flights": FLIGHTS_GOAL, "hotels-fixture": FIXTURE_GOAL}


def wide_page(base, extra):
    """Synthetic: a real page plus `extra` generic links, to exercise coarse-to-fine target selection."""
    page = json.loads(json.dumps(base))
    scroll = [a for a in page["actions"] if a["kind"] in {"scroll", "wait"}]
    page["actions"] = [a for a in page["actions"] if a["kind"] not in {"scroll", "wait"}]
    page["actions"] += [
        {"id": f"x{i}", "kind": "click", "role": "link", "label": f"Result {i}: flight option {i}", "value": "",
         "node": 1000 + i}
        for i in range(1, extra + 1)
    ]
    for n, action in enumerate(page["actions"], 1):
        action["id"] = f"e{n}"
    page["actions"] += scroll
    return page


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def run_page(decider, case, repeats):
    page, goal = case["page"], GOALS[case["task"]]
    latencies, last = [], None
    for _ in range(repeats):
        started = time.perf_counter()
        try:
            last = model.choose(page, goal, [], decider)
        except ValueError as refusal:  # a refused decision is a result: no action would have executed
            return {"page": case["name"], "elements": len(model.action_space(page["actions"])[0]),
                    "operation": "REFUSED", "chosen": str(refusal), "plausible": False, "model_calls": None,
                    "truncated": None, "first_ms": None, "steady_median_ms": None, "steady_p95_ms": None}
        latencies.append((time.perf_counter() - started) * 1000)
    control = {"label": last["choice"], "kind": "control"}
    action = next((a for a in page["actions"] if a["id"] == last["choice"]), control)
    ok_ops, ok_labels = case.get("ok_operations", []), case.get("ok_labels", [])
    plausible = None
    if ok_ops or ok_labels:
        plausible = last["operation"] in ok_ops or any(label in action["label"] for label in ok_labels)
    return {
        "page": case["name"],
        "elements": len(model.action_space(page["actions"])[0]),
        "operation": last["operation"],
        "chosen": action["label"],
        "plausible": plausible,
        "operation_confidence": last["confidence"],
        "model_calls": last["model_calls"],
        "input_tokens": last["usage"].get("input_tokens"),
        "truncated": bool(last["info"].get("truncated")),
        "split_heads": last["info"].get("split_heads", []),
        "first_ms": round(latencies[0], 1),
        "steady_median_ms": round(statistics.median(latencies[1:]), 1) if repeats > 1 else None,
        "steady_p95_ms": round(percentile(latencies[1:], 0.95), 1) if repeats > 2 else None,
        "steady_min_ms": round(min(latencies[1:]), 1) if repeats > 1 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--wide", type=int, default=150, help="Extra synthetic links for a wide page (0 = skip).")
    parser.add_argument("--output")
    args = parser.parse_args()
    cases = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "benchmarks" / "pages").glob("*.json"))]
    if args.wide:
        base = next(c for c in cases if c["name"] == "flights_home")
        cases.append({"name": f"flights_home + {args.wide} synthetic links", "task": "flights",
                      "page": wide_page(base["page"], args.wide)})
    os.environ.setdefault("LAYA_PRELOAD", "0")
    decider = decision_model_from_env()
    started = time.perf_counter()
    decider.load()
    wall_load_ms = round((time.perf_counter() - started) * 1000)
    import laya
    import torch

    environment = {
        "checkpoint": decider.model, "kind": decider.kind, "device": str(decider.agent.device),
        "revision": getattr(decider.agent, "revision", None) or decider.revision,
        "max_len": decider.budget["max_len"] or decider.agent.cfg.get("max_len"),
        "head_max_len": decider.budget["head_max_len"] or decider.agent.cfg.get("head_max_len"),
        "max_options": decider.max_options, "laya": getattr(laya, "__version__", None), "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(), "cpu": platform.processor(), "cores": os.cpu_count(),
        "load_ms": decider.load_ms, "warmup_ms": decider.warmup_ms, "wall_load_plus_warmup_ms": wall_load_ms,
        "repeats": args.repeats,
    }
    rows = [run_page(decider, case, args.repeats) for case in cases]
    print(json.dumps(environment, indent=1))
    print("| page | elements | operation | chosen | plausible | calls | truncated | first ms | steady ms | p95 ms |")
    print("| --- | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |")
    for r in rows:
        print(
            f"| {r['page']} | {r['elements']} | {r['operation']} | {r['chosen'][:42]} | {r['plausible']} | "
            f"{r['model_calls']} | {r['truncated']} | {r['first_ms']} | {r['steady_median_ms']} | "
            f"{r['steady_p95_ms']} |"
        )
    if args.output:
        report = json.dumps({"environment": environment, "rows": rows}, indent=1)
        Path(args.output).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
