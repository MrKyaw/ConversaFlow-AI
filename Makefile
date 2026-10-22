.PHONY: install train eval test lint typecheck serve docker clean

install:
	uv venv && uv pip install -e ".[dev]"

train:
	conversaflow train

eval:
	conversaflow eval --split test --markdown docs/TEST_RESULTS.md

test:
	pytest --cov=src/conversaflow --cov-fail-under=80

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

typecheck:
	mypy src

serve:
	conversaflow serve

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache var models/nlu.json
