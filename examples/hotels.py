"""A hotel-search task on the same agent as flights. Nothing in the decision layer changes.

Two tasks live here:

* `FIXTURE` runs against the bundled local stays page, with a strict independent check.
* `search(...)` builds a goal for a real hotel site. Its check is deliberately conservative: it proves the
  page is not a checkout and echoes the requested city; it cannot prove a site's filters were applied.

Neither enters payment details or books anything.

    uv run --env-file .env laya-ultrafast          # serves the fixture page on 127.0.0.1:8766
    uv run --env-file .env python examples/hotels.py --fixture
    uv run --env-file .env python examples/hotels.py --url https://hotels.example --city Stockholm \\
        --dates "14-18 November" --budget "under 800 GBP total" --rating "above 8" \\
        --require "private bathroom" --require "free cancellation"
"""

import argparse
import json
import re
from pathlib import Path

from laya_ultrafast import Agent, Task, run_task

FIXTURE_URL = "http://127.0.0.1:8766/fixture.html?scenario=travel"
FIXTURE_GOAL = (
    "Use the destination search and filters to find Design stays in Lisbon with Free cancellation, "
    "then open Casa Flora. Do not book anything."
)
CHECKOUT = re.compile(
    r"card number|cvv|cvc|security code|expiry|billing address|pay now|confirm (and pay|booking)", re.I
)


def verify_fixture(page):
    checks = {
        "opened_casa_flora": page["url"].endswith("#casa-flora"),
        "design_filter": "Your filters: Design" in page["text"],
        "free_cancellation": "Free cancellation enabled" in page["text"],
        "destination": "Destination Lisbon" in page["text"],
        "no_booking": not CHECKOUT.search(page["text"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


FIXTURE = Task("hotels-fixture", FIXTURE_URL, FIXTURE_GOAL, verify_fixture)


def search(url, city, dates, budget=None, rating=None, require=()):
    """A task for a real hotel site. Requirements are goal text, never hardcoded policy."""
    parts = [f"Find a hotel in {city} for {dates}"]
    parts += [f"budget {budget}"] if budget else []
    parts += [f"rating {rating}"] if rating else []
    parts += [f"must have {r}" for r in require]
    goal = (
        ", ".join(parts)
        + ". Search, set the dates, guests, filters and sort order, then open the best matching property page. "
        "Do not book, reserve, sign in, or enter payment details."
    )

    def verify(page):
        checks = {
            "no_booking_or_payment_page": not CHECKOUT.search(page["text"]),
            "mentions_city": city.lower() in page["text"].lower() or city.lower() in page["url"].lower(),
        }
        return {"passed": all(checks.values()), "checks": checks, "note": "Filters are not independently provable."}

    return Task("hotels", url, goal, verify)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--url")
    parser.add_argument("--city")
    parser.add_argument("--dates")
    parser.add_argument("--budget")
    parser.add_argument("--rating")
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--output", default="artifacts/hotels/latest")
    args = parser.parse_args()
    if args.fixture:
        task = FIXTURE
    elif args.url and args.city and args.dates:
        task = search(args.url, args.city, args.dates, args.budget, args.rating, args.require)
    else:
        parser.error("Use --fixture, or supply --url, --city and --dates")
    agent = Agent(task.url, task.goal)
    print(f"Decision model: {agent.decider.description}", flush=True)

    def progress(state):
        print(state["elapsed_ms"], state["status"], state["history"][-1]["action"] if state["history"] else "")

    result = run_task(task, agent=agent, progress=progress)
    agent.close()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "state.json").write_text(json.dumps(result["state"], indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "passed", "verification")}, indent=2))
    if not result["passed"]:
        raise SystemExit(result["error"] or "Independent verification failed")


if __name__ == "__main__":
    main()
