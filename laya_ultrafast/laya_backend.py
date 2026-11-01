"""In-process Laya as the decision model: typed questions in, calibrated typed answers out.

One `system_one` forward pass answers the operation question and every operation's target head together.
Laya's option budget is small, so a head wider than `max_options` is answered coarse-to-fine: interleaved
chunks in the same pass, then only the chunk winners compete in one more pass. Nothing here touches the
browser, and nothing here is trusted: `model.choose` validates every answer before an action can exist.

The compact request format follows the one `cklxx/laya-browser` was fine-tuned on (Apache-2.0,
https://huggingface.co/cklxx/laya-browser, `laya_browser.py`). It is only a compression of the same
observed state, so the generic checkpoint receives it too.
"""

import os
import re
import threading
import time

# Third-party browser fine-tune of Laya (Apache-2.0), pinned to the commit that was inspected and measured.
# Generic `convaiinnovations/laya` is selectable, but it answered DONE/BLOCKED on every recorded page
# (docs/performance-laya.md).
DEFAULT_MODEL = "cklxx/laya-browser"
DEFAULT_REVISION = "645cf366a2ae35f1086e8c20eff48f909bb49206"
PAGE_TEXT_CHARS = 1200
LABEL_CHARS = 50
BROWSER_MAX_OPTIONS = 60  # what the browser fine-tune was trained to read in one head
GENERIC_MAX_OPTIONS = 12  # generic Laya's default 192-token head budget collapses wider heads
STATE_FLAGS = ("checked", "selected", "expanded")


def _cut(text, limit):
    text = str(text)
    if len(text) <= limit:
        return text
    if " → " in text:  # a <select> option keeps its option name
        head, option = text.rsplit(" → ", 1)
        option = " ".join(option.split())[:40]
        return " ".join(head.split())[: max(12, limit - len(option) - 3)] + " → " + option
    return text[:limit]


def compact_option(option):
    """`Field → Option` for a dropdown value; otherwise `label (role) = 'value' flag=...`."""
    if not isinstance(option, dict) or "element" not in option:
        return option
    text = re.sub(r"^\[[^\]]*\]\s*", "", str(option["element"]))
    if " → " in text:
        return _cut(text, LABEL_CHARS)
    text = _cut(text, LABEL_CHARS)
    if option.get("role"):
        text += f" ({option['role']})"
    if option.get("current_value"):
        text += f" = {str(option['current_value'])[:30]!r}"
    return text + "".join(f" {k}={option[k]}" for k in STATE_FLAGS if k in option)


def fields_summary(elements):
    """Every form field with its current value, first in the state so truncation never drops it."""
    out = []
    for e in elements or []:
        operations, role = e.get("operations") or [], e.get("role")
        label = str(e.get("label", ""))[:40]
        if "TYPE_TEXT" in operations or "SELECT" in operations or role == "combobox":
            value = str(e.get("value") or "").strip()
            out.append(f"{label} = {value[:30]!r}" if value else f"{label} = (empty)")
        elif role in ("checkbox", "radio", "switch") and "checked" in e:
            out.append(f"{label}: checked={e['checked']}")
        if len(out) >= 14:
            break
    return "; ".join(out)


def split_wide(questions, max_options):
    """Replace each choice wider than `max_options` with interleaved chunk questions."""
    asked, plan = {}, {}
    for qid, question in questions.items():
        keys = list(question["criteria"]) if question.get("type") == "choice" else []
        if len(keys) <= max_options:
            asked[qid] = question
            continue
        count = -(-len(keys) // max_options)
        chunks = [keys[i::count] for i in range(count)]
        plan[qid] = (question, chunks)
        for i, chunk in enumerate(chunks):
            asked[f"{qid}__chunk{i}"] = {**question, "criteria": {k: question["criteria"][k] for k in chunk}}
    return asked, plan


def merge_chunks(answers, plan, final):
    """p(option) = p(its chunk's winner among winners) * p(option within its chunk), renormalised."""
    for qid, (_question, chunks) in plan.items():
        winners = [answers.pop(f"{qid}__chunk{i}") for i in range(len(chunks))]
        probabilities = {}
        for winner, chunk in zip(winners, chunks):
            for key in chunk:
                probabilities[key] = final[qid]["probabilities"][winner["choice"]] * winner["probabilities"][key]
        total = sum(probabilities.values()) or 1.0
        probabilities = {k: v / total for k, v in probabilities.items()}
        answers[qid] = {
            "type": "choice",
            "choice": max(probabilities, key=probabilities.get),
            "probabilities": probabilities,
            "confidence": final[qid]["confidence"],
        }


class LayaDecisionModel:
    """Loads one Laya checkpoint once and reuses it for every decision in the process."""

    def __init__(
        self,
        model=DEFAULT_MODEL,
        device="auto",
        *,
        revision=None,
        max_len=None,
        head_max_len=None,
        max_options=None,
        page_text_chars=None,
        loader=None,
    ):
        self.model, self.revision = model, revision or None
        self.device = None if str(device or "auto").lower() == "auto" else device
        self.budget = {"max_len": max_len, "head_max_len": head_max_len}
        self.max_options, self.page_text_chars = max_options, page_text_chars
        self.loader = loader
        self.agent = None
        self.kind = None
        self.load_ms = 0
        self.warmup_ms = 0
        self._lock = threading.Lock()

    @property
    def description(self):
        if self.agent is None:
            return f"{self.model} (not loaded)"
        return f"{self.model} · {self.kind} · {self.agent.device}"

    def load(self):
        """Load and warm up once. Cost is recorded here and never charged to a decision."""
        with self._lock:
            if self.agent is not None:
                return 0
            loader = self.loader
            if loader is None:
                import laya  # imported lazily: tests and tooling do not need torch

                loader = laya.load
            started = time.perf_counter()
            try:
                agent = loader(self.model, device=self.device, revision=self.revision)
            except Exception as error:
                raise RuntimeError(
                    f"Could not load Laya checkpoint {self.model!r}: {error}. "
                    "Check LAYA_MODEL, network access for the first download, and LAYA_DEVICE."
                ) from error
            self.load_ms = round((time.perf_counter() - started) * 1000)
            config = getattr(agent, "cfg", {})
            # Browser fine-tunes record the request format they were trained on; the generic checkpoint does not.
            self.kind = "browser-fine-tuned" if config.get("laya_fmt") else "generic"
            if self.max_options is None:
                self.max_options = BROWSER_MAX_OPTIONS if self.kind == "browser-fine-tuned" else GENERIC_MAX_OPTIONS
            if self.page_text_chars is None:
                self.page_text_chars = PAGE_TEXT_CHARS
            self.agent = agent
            started = time.perf_counter()
            self._predict(
                {"page": {"url": "", "title": "", "text": ""}},
                {"warmup": {"type": "choice", "criteria": {"a": "a", "b": "b"}, "instructions": "warm up"}},
            )
            self.warmup_ms = round((time.perf_counter() - started) * 1000)
            return self.load_ms

    def _predict(self, state, questions):
        overrides = {k: v for k, v in self.budget.items() if v is not None}
        return self.agent.system_one(state, questions, **overrides)

    def compact_state(self, state):
        page = state.get("page", {})
        return {
            "fields": fields_summary(state.get("elements")),
            "page": {**page, "text": (page.get("text") or "")[: self.page_text_chars]},
            "recent_actions": state.get("recent_actions", [])[-10:],
        }

    def choose(self, state, questions):
        load_ms = self.load()
        questions = {
            qid: {**q, "criteria": {k: compact_option(v) for k, v in q["criteria"].items()}}
            if isinstance(q.get("criteria"), dict)
            else q
            for qid, q in questions.items()
        }
        compact = self.compact_state(state)
        asked, plan = split_wide(questions, self.max_options)
        first = self._predict(compact, asked)
        answers, results = first["answers"], [first]
        if plan:
            winners = {
                qid: [answers[f"{qid}__chunk{i}"]["choice"] for i in range(len(chunks))]
                for qid, (_question, chunks) in plan.items()
            }
            finals = {
                qid: {**question, "criteria": {w: question["criteria"][w] for w in winners[qid]}}
                for qid, (question, _chunks) in plan.items()
            }
            second = self._predict(compact, finals)
            results.append(second)
            merge_chunks(answers, plan, second["answers"])
        usage = {"input_tokens": sum(r.get("usage", {}).get("input_tokens", 0) for r in results), "output_tokens": 0}
        collapsed = {
            qid.split("__chunk")[0] for r in results for qid in r.get("usage", {}).get("options", {})
        }
        return {
            "answers": answers,
            "model": self.model,
            "kind": self.kind,
            "usage": usage,
            "model_calls": len(results),
            "info": {
                "load_ms": load_ms,
                "truncated": any(r.get("usage", {}).get("truncated") for r in results),
                "collapsed": sorted(collapsed),
                "revision": getattr(self.agent, "revision", None) or self.revision,
                "split_heads": sorted(plan),
            },
        }


_MODELS = {}


def decision_model_from_env(environ=None, loader=None):
    """One shared instance per configuration, so every Agent in a process reuses the loaded checkpoint."""
    env = os.environ if environ is None else environ

    def number(name):
        return int(env[name]) if env.get(name) else None

    model = env.get("LAYA_MODEL") or DEFAULT_MODEL
    config = (
        model,
        env.get("LAYA_DEVICE") or "auto",
        env.get("LAYA_REVISION") or (DEFAULT_REVISION if model == DEFAULT_MODEL else None),
        number("LAYA_MAX_LEN"),
        number("LAYA_HEAD_MAX_LEN"),
        number("LAYA_MAX_OPTIONS"),
        number("LAYA_PAGE_TEXT_CHARS"),
    )
    key = (config, id(loader))
    if key not in _MODELS:
        _, device, revision, max_len, head_max_len, max_options, page_text_chars = config
        _MODELS[key] = LayaDecisionModel(
            model,
            device,
            revision=revision,
            max_len=max_len,
            head_max_len=head_max_len,
            max_options=max_options,
            page_text_chars=page_text_chars,
            loader=loader,
        )
    decider = _MODELS[key]
    if env.get("LAYA_PRELOAD", "1").lower() not in {"0", "false", "no", ""}:
        decider.load()
    return decider
