"""Deterministic stand-ins. The decision model and the text helper are separate fakes on purpose."""


def pick(criteria, selected, confidence=1.0):
    """A valid one-hot Laya-shaped answer over `criteria`."""
    return {
        "type": "choice",
        "choice": selected,
        "confidence": confidence,
        "probabilities": {k: float(k == selected) for k in criteria},
    }


class FakeDecider:
    """Answers each question with whatever `script(question_id, question)` returns; records every call."""

    description = "fake · generic · cpu"
    load_ms = 0
    warmup_ms = 0

    def __init__(self, script, info=None, model_calls=1):
        self.script, self.info, self.model_calls = script, info or {}, model_calls
        self.calls = []

    def choose(self, state, questions):
        self.calls.append((state, questions))
        answers = {}
        for qid, question in questions.items():
            answer = self.script(qid, question)
            if answer is not None:
                answers[qid] = answer
        return {
            "answers": answers,
            "model": "fake",
            "kind": "generic",
            "usage": {},
            "model_calls": self.model_calls,
            "info": self.info,
        }


def choosing(operation, target=None):
    """A script that picks `operation` and, when given, `target` in that operation's head."""

    def script(qid, question):
        if qid == "operation":
            return pick(question["criteria"], operation)
        if target is not None and qid == operation.lower() + "_target":
            return pick(question["criteria"], target)
        return None

    return script
