.PHONY: setup corpus golden index eval api ui test lint docker-build docker-up clean

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install -e ".[dev]"

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
