.PHONY: setup corpus golden index eval api ui test lint docker-build docker-up clean \
        agent-queue agent-run agent-drain agent-tools agent-mcp agent-eval

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install -e ".[dev,agent]"

corpus:
	python scripts/generate_corpus.py

golden:
	python scripts/build_golden_set.py

index:
	python scripts/build_index.py

eval:
	python scripts/run_eval.py

api:
	uvicorn ragcite.api.main:app --app-dir src --reload --port 8000

ui:
	streamlit run src/ragcite/app/streamlit_app.py

# --- Project B: reconciliation agent ---------------------------------------

EXC ?= EXC-0001

agent-queue:
	python -m reconagent.cli queue

agent-run:
	python -m reconagent.cli run $(EXC)

agent-drain:
	python -m reconagent.cli drain --limit 10

agent-tools:
	python -m reconagent.cli tools

agent-mcp:
	python scripts/mcp_server.py

agent-eval:
	python scripts/run_agent_eval.py

test:
	pytest -q

lint:
	ruff check src tests scripts

docker-build:
	docker compose build

docker-up:
	docker compose up

clean:
	rm -rf data/index .pytest_cache .ruff_cache
