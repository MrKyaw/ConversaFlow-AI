"""Command-line interface for ConversaFlow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import __version__
from .config import Settings
from .nlu.train import save_model, train_from_file
from .runtime.bootstrap import build_runtime


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def cmd_train(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    train_path = data_dir / "intents" / "train.jsonl"
    model, report = train_from_file(train_path)
    output = Path(args.output)
    save_model(model, output)
    _print_json(
        {
            "model_path": str(output),
            "train_accuracy": report.train_accuracy,
            "validation_accuracy": report.validation_accuracy,
            "epochs": report.epochs,
            "temperature": report.temperature,
            "class_counts": report.class_counts,
        }
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation.harness import evaluate_conversations, evaluate_intents, to_dict

    data_dir = Path(args.data_dir)
    split = args.split
    corpus = data_dir / "intents" / f"{split}.jsonl"
    settings = Settings.from_env(
        {
            "CONVERSAFLOW_STORE": "memory",
            "CONVERSAFLOW_MODEL_PATH": args.model_path or "models/nlu.json",
            "CONVERSAFLOW_DATA_DIR": str(data_dir),
        }
    )
    runtime = build_runtime(settings)
    intent_report = evaluate_intents(runtime.nlu, corpus)
    golden_dir = Path(args.golden_dir)
    conversation_report = None
    if golden_dir.exists():
        conversation_report = evaluate_conversations(runtime, golden_dir)

    summary = {
        "split": split,
        "intent": {
            "total": intent_report.total,
            "correct": intent_report.correct,
            "accuracy": intent_report.accuracy,
            "macro_f1": intent_report.macro_f1,
            "mean_confidence": intent_report.mean_confidence,
            "mean_margin": intent_report.mean_margin,
            "low_confidence_rate": intent_report.low_confidence_rate,
        },
    }
    if conversation_report is not None:
        summary["conversations"] = {
            "total": conversation_report.total,
            "successful": conversation_report.successful,
            "success_rate": conversation_report.success_rate,
            "containment_rate": conversation_report.containment_rate,
            "average_turns": conversation_report.average_turns,
        }
    _print_json(summary)

    if args.json:
        payload = {
            "intent": to_dict(intent_report),
            "conversations": to_dict(conversation_report) if conversation_report else None,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if args.markdown:
        _write_markdown(args.markdown, split, intent_report, conversation_report)
    return 0


def _write_markdown(path: str, split: str, intent_report: Any, conversation_report: Any) -> None:
    lines = [
        f"# Evaluation report ({split})",
        "",
        "## Intent classification",
        "",
        f"- Total examples: {intent_report.total}",
        f"- Correct: {intent_report.correct}",
        f"- Accuracy: {intent_report.accuracy:.4f}",
        f"- Macro F1: {intent_report.macro_f1:.4f}",
        f"- Mean confidence: {intent_report.mean_confidence:.4f}",
        f"- Mean margin: {intent_report.mean_margin:.4f}",
        f"- Low-confidence rate: {intent_report.low_confidence_rate:.4f}",
        "",
        "| Intent | Support | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in sorted(intent_report.per_intent.items()):
        lines.append(
            f"| {name} | {metrics['support']} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1']:.4f} |"
        )
    if conversation_report is not None:
        lines.extend(
            [
                "",
                "## Conversations",
                "",
                f"- Total: {conversation_report.total}",
                f"- Successful: {conversation_report.successful}",
                f"- Success rate: {conversation_report.success_rate:.4f}",
                f"- Containment rate: {conversation_report.containment_rate:.4f}",
                f"- Average turns: {conversation_report.average_turns:.2f}",
            ]
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_simulate(args: argparse.Namespace) -> int:
    from .models import TurnRequest

    data_dir = Path(args.data_dir)
    settings = Settings.from_env(
        {
            "CONVERSAFLOW_STORE": "memory",
            "CONVERSAFLOW_MODEL_PATH": args.model_path or "models/nlu.json",
            "CONVERSAFLOW_DATA_DIR": str(data_dir),
        }
    )
    runtime = build_runtime(settings)
    import asyncio

    texts = list(args.texts or [])
    if args.script:
        script_path = Path(args.script)
        texts = [line.strip() for line in script_path.read_text().splitlines() if line.strip()]

    async def run() -> None:
        session = await runtime.engine.create_session("demo")
        for text in texts:
            response = await runtime.engine.handle_turn(
                "demo", TurnRequest(text=text, session_id=session.id)
            )
            print(f"USER: {text}")
            print(f"BOT : {response.text}")
            print(
                f"      intent={response.intent} "
                f"conf={response.confidence:.2f} flow={response.dialogue.flow}"
            )
            session = await runtime.engine.get_session(session.id, "demo")

    asyncio.run(run())
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "conversaflow.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower(),
    )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    settings = Settings.from_env(
        {
            "CONVERSAFLOW_STORE": "memory",
            "CONVERSAFLOW_MODEL_PATH": args.model_path or "models/nlu.json",
            "CONVERSAFLOW_DATA_DIR": str(data_dir),
        }
    )
    runtime = build_runtime(settings)
    _print_json(
        {
            "version": __version__,
            "store": settings.store,
            "model_path": str(settings.model_path),
            "data_dir": str(settings.data_dir),
            "health": runtime.engine.health().model_dump(mode="json"),
            "flows": [flow.id for flow in runtime.flows.values()],
            "agents": [agent["name"] for agent in runtime.router.catalogue()],
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conversaflow", description="ConversaFlow AI runtime")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train the intent model")
    train.add_argument("--data-dir", default="data")
    train.add_argument("--output", default="models/nlu.json")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("eval", help="Evaluate NLU and conversations")
    evaluate.add_argument("--split", choices=["train", "eval", "test"], default="test")
    evaluate.add_argument("--data-dir", default="data")
    evaluate.add_argument("--golden-dir", default="data/golden")
    evaluate.add_argument("--model-path", default=None)
    evaluate.add_argument("--json", default=None)
    evaluate.add_argument("--markdown", default=None)
    evaluate.set_defaults(func=cmd_eval)

    simulate = subparsers.add_parser("simulate", help="Simulate a conversation")
    simulate.add_argument("texts", nargs="*")
    simulate.add_argument("--script", default=None)
    simulate.add_argument("--data-dir", default="data")
    simulate.add_argument("--model-path", default=None)
    simulate.set_defaults(func=cmd_simulate)

    serve = subparsers.add_parser("serve", help="Run the HTTP server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    info = subparsers.add_parser("info", help="Print runtime information")
    info.add_argument("--data-dir", default="data")
    info.add_argument("--model-path", default=None)
    info.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
