# Changelog

All notable changes to this project are documented in this file.

## [0.2.0] - Unreleased

### Added
- **Java language support** — `husk scan`/`graph`/`hotspots`/`deadcode`/`doc` now understand `.java` files (classes, interfaces, enums, methods/constructors, imports, cyclomatic complexity), alongside the existing Python/JS/TS support.
- **Zero-API-key mode** — `husk init` now offers a `local` provider that auto-provisions a small default model (`qwen2.5-coder:1.5b` for chat/summarization, `nomic-embed-text` for embeddings) via Ollama, so Husk works without any paid API key as long as Ollama is installed. Commands auto pull missing models on first use.
- **Colored CLI menu** — running `husk` with no arguments (or `husk help`) now shows a styled command menu instead of Click's default listing.
- `CHANGELOG.md`. (CI already existed via `.github/workflows/test.yml`, which now also covers the `tree-sitter-java` dependency through `requirements.txt`.)

### Changed
- `husk/parser.py` and `husk/graph.py` were refactored from per-language `if/elif` chains into a data-driven `LanguageSpec` table (`husk/langspec.py`), making it straightforward to add further languages (C++ planned next).
- `husk ask`'s RAG index is now incrementally updated (SHA256 content hashing per file, like the existing summary cache) instead of an all-or-nothing rebuild; `--rebuild` remains as an explicit full re-index.
- Oversized code chunks (>120 lines) are now split into overlapping sub-chunks before embedding, instead of being sent as one unbounded chunk.
- Query embeddings for `husk ask` are cached on disk, so repeated/iterative questions don't re-pay for the same embedding call.
- Model pricing table in `husk/ai/estimator.py` updated with current OpenAI/Anthropic model IDs.

### Fixed
- `husk ask` no longer crashes for users on the `anthropic` provider (Anthropic has no embeddings API) — it now transparently falls back to local Ollama embeddings with a clear message instead of raising an unhandled error.
- Removed an unreachable/dead branch in `DependencyGraphBuilder.resolve_python_import`.
