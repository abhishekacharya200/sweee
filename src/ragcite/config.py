from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "mock"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None)
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"))
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or None)
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "local"))
    sentence_transformers_model: str = field(
        default_factory=lambda: os.getenv(
            "SENTENCE_TRANSFORMERS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    rerank_backend: str = field(default_factory=lambda: os.getenv("RERANK_BACKEND", "lexical"))
    cross_encoder_model: str = field(
        default_factory=lambda: os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    top_k_sparse: int = field(default_factory=lambda: int(os.getenv("TOP_K_SPARSE", "15")))
    top_k_dense: int = field(default_factory=lambda: int(os.getenv("TOP_K_DENSE", "15")))
    top_k_fused: int = field(default_factory=lambda: int(os.getenv("TOP_K_FUSED", "10")))
    top_k_final: int = field(default_factory=lambda: int(os.getenv("TOP_K_FINAL", "5")))

    corpus_dir: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("CORPUS_DIR", "data/corpus"))
    index_dir: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("INDEX_DIR", "data/index"))

    chunk_max_tokens: int = 60
    chunk_overlap_paragraphs: int = 0

    def resolve_llm_provider(self) -> str:
        """Fall back to mock if the configured provider has no key set."""
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            return "mock"
        if self.llm_provider == "openai" and not self.openai_api_key:
            return "mock"
        return self.llm_provider


settings = Settings()
