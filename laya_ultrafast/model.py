"""Operation + target decisions. A DecisionModel answers typed questions; this module validates every answer."""

import math
import time
from typing import Protocol

from .questions import NEXT_ACTION, TARGET


class DecisionModel(Protocol):
    """The whole integration boundary: typed questions in, typed answers out.

    `choose` returns `{"answers": {question_id: {"choice", "probabilities", "confidence"}}, "model": str, ...}`
    and may add `usage`, `model_calls`, and `info`. It must not mutate the browser; its answers are untrusted
    until `validate_choice` accepts them.
    """

    def choose(self, state: dict, questions: dict) -> dict: ...


def validate_choice(answer, ids):
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        valid = False
    if not valid:
        raise ValueError("Invalid decision response; no action executed.")
    return answer


def action_space(actions):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


def build_request(page, goal, history):
    """The backend-neutral request: observed state plus one operation question and one head per operation."""
    elements, targets, controls = action_space(page["actions"])
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
    }
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {
                index: {
                    "element": f"[{index}] {a['label']}",
                    "current_value": a.get("current_value", a.get("value", "")),
                    **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
                }
                for index, a in candidates.items()
            },
            "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        }
    state = {
        "page": {k: page[k] for k in ("url", "title", "text")},
        "elements": elements,
        "recent_actions": [{k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]],
    }
    return state, questions, targets, controls, operations


def choose(page, goal, history, decider):
    state, questions, targets, controls, operations = build_request(page, goal, history)
    started = time.perf_counter()
    result = decider.choose(state, questions)
    total_ms = round((time.perf_counter() - started) * 1000)
    answers = result.get("answers") if isinstance(result, dict) else None
    if not isinstance(answers, dict):
        raise ValueError("Invalid decision response; no action executed.")
    # A lazy first-call model load is reported on its own; it is not a decision's latency.
    load_ms = result.get("info", {}).get("load_ms", 0)
    operation_answer = validate_choice(answers.get("operation", {}), operations)
    operation = operation_answer["choice"]
    # Options that lost their own token span cannot be told apart by the model; refuse rather than guess among them.
    collapsed = set(result.get("info", {}).get("collapsed", ()))
    if collapsed & {"operation", operation.lower() + "_target"}:
        raise ValueError("Decision options exceeded the model's budget; no action executed.")
    target = None
    target_answer = None
    probabilities = {}
    if operation in targets:
        # Unused target heads cannot cause an action. Validate the head selected by the operation.
        target_answer = validate_choice(answers.get(operation.lower() + "_target", {}), targets[operation])
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": answers,
        "model": result.get("model"),
        "model_kind": result.get("kind"),
        "usage": result.get("usage", {}),
        "model_calls": result.get("model_calls", 1),
        "info": result.get("info", {}),
        "latency_ms": total_ms - load_ms,
        "load_ms": load_ms,
        "request": {"state": state, "questions": questions},
    }
