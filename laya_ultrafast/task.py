"""A task is data: a start URL, a natural-language goal, and an independent check of the final page.

Flights and hotels are tasks, not agents. Nothing here reaches the decision model.
"""

from dataclasses import dataclass
from typing import Callable

from .agent import Agent


@dataclass(frozen=True)
class Task:
    name: str
    url: str
    goal: str
    verify: Callable[[dict], dict]  # final observed page -> {"passed": bool, "checks": {...}, ...}


def run_task(task, agent=None, progress=None, **agent_kwargs):
    """Run a task, then verify the real page. A DONE choice is a claim, never the result."""
    owns = agent is None
    agent = agent or Agent(task.url, task.goal, **agent_kwargs)
    error = None
    try:
        for state in agent.run():
            if progress:
                progress(state)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    state = agent.snapshot()
    try:
        # A fresh observation, not the page the agent last saw.
        final = agent.browser.observe(screenshot=False)
        verification = task.verify(final)
    except Exception as exc:
        final = None
        verification = {"passed": False, "checks": {}, "error": f"{type(exc).__name__}: {exc}"}
    if owns:
        agent.close()
    return {
        "task": task.name,
        "status": state["status"],
        "error": error,
        "passed": bool(verification["passed"]),
        "verification": verification,
        "state": state,
        "final_page": final,
    }
