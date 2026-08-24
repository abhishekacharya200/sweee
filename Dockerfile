FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY eval ./eval
RUN pip install --no-cache-dir -e .

# Corpus + eval golden set ship in the image so the container is runnable
# with zero setup; the index is built on first boot (see entrypoint).
COPY data ./data

ENV PYTHONUNBUFFERED=1
EXPOSE 8000 8501

FROM base AS api
CMD ["uvicorn", "ragcite.api.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS ui
CMD ["streamlit", "run", "src/ragcite/app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
