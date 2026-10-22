"""Run a canned multi-turn demo conversation against an in-memory runtime."""

from __future__ import annotations

from conversaflow.config import Settings
from conversaflow.models import TurnRequest
from conversaflow.runtime.bootstrap import build_runtime

SCRIPT = [
    "Hi there",
    "I want a refund for order ORD-1234 because it arrived broken",
    "yes",
    "where is my order ORD-7788",
    "I want to book an appointment",
    "repair",
    "2026-11-10",
    "3pm",
    "yes",
    "Do you support single sign-on?",
    "no, that's all",
]


def main() -> None:
    settings = Settings.from_env(
        {"CONVERSAFLOW_STORE": "memory", "CONVERSAFLOW_MODEL_PATH": "var/demo-model.json"}
    )
    runtime = build_runtime(settings)
    import asyncio

    async def run() -> None:
        session = await runtime.engine.create_session("demo")
        for text in SCRIPT:
            response = await runtime.engine.handle_turn(
                "demo", TurnRequest(text=text, session_id=session.id)
            )
            print(f"USER: {text}")
            print(f"BOT : {response.text}\n")
            session = await runtime.engine.get_session(session.id, "demo")

    asyncio.run(run())


if __name__ == "__main__":
    main()
