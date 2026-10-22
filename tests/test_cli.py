"""Tests for the CLI."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_version():
    result = subprocess.run(
        [sys.executable, "-m", "conversaflow", "--version"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0
    assert "1.0.0" in result.stdout


def test_cli_info():
    result = subprocess.run(
        [sys.executable, "-m", "conversaflow", "info"], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0


def test_cli_simulate():
    result = subprocess.run(
        [sys.executable, "-m", "conversaflow", "simulate", "hello", "thanks"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0
    assert "BOT :" in result.stdout


def test_cli_eval(tmp_path):
    out = tmp_path / "report.json"
    md = tmp_path / "report.md"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "conversaflow",
            "eval",
            "--split",
            "eval",
            "--json",
            str(out),
            "--markdown",
            str(md),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0
    assert out.exists() and md.exists()


def test_cli_train(tmp_path):
    out = tmp_path / "model.json"
    result = subprocess.run(
        [sys.executable, "-m", "conversaflow", "train", "--output", str(out)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0
    assert out.exists()


def test_cli_commands_in_process(tmp_path, capsys, monkeypatch):
    from conversaflow.cli import main

    assert main(["info"]) == 0
    assert '"version": "1.0.0"' in capsys.readouterr().out
    report = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    assert main(["eval", "--json", str(report), "--markdown", str(markdown)]) == 0
    assert report.exists() and markdown.exists()
    assert main(["simulate", "hello", "thanks"]) == 0
    assert "USER: hello" in capsys.readouterr().out
    script = tmp_path / "conversation.txt"
    script.write_text("hello\n\nthanks\n")
    assert main(["simulate", "--script", str(script)]) == 0

    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert main(["serve", "--host", "127.0.0.1", "--port", "8123"]) == 0
    assert calls[0][1]["port"] == 8123


def test_cli_train_in_process(tmp_path):
    import json

    from conversaflow.cli import main

    intents = tmp_path / "intents"
    intents.mkdir()
    rows = [
        {"text": text, "intent": intent}
        for intent, texts in {
            "greet": ["hello", "hi", "hey"],
            "bye": ["bye", "goodbye", "later"],
        }.items()
        for text in texts
    ]
    (intents / "train.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "trained.json"
    assert main(["train", "--data-dir", str(tmp_path), "--output", str(output)]) == 0
    assert set(json.loads(output.read_text())["classes"]) == {"greet", "bye"}
