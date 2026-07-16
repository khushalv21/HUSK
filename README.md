# Husk

*Your codebase's past, present, and architecture—laid bare.*

Understand legacy codebases quickly and privately with offline static analysis and local LLMs, without uploading code or fighting heavy setups.

---

## ❓ Why Husk?
> Because code archaeology should be secure, local, and instant — without SaaS lock-in or platform-specific vector databases.

---

## 🚀 Key Features

* **Static Analysis (Tier 1):** Code complexity, git churn, hotspots rankings, dead code detection, and Mermaid dependency graph exports.
* **Semantic AI (Tier 2):** Hierarchical Map-Reduce documentation generator, token/budget limits, and natural language Q&A index with Git Blame annotations.
* **Remote Git Scanning:** Provide a Git clone link (`https://github.com/...`) instead of a local directory, and Husk will automatically download and scan it temporarily.
* **Local Offline AI:** Fully integrates with local Ollama servers (like `llama3`) for offline, zero-cost semantic search and summarization.

---

## 📦 Quick Start

```bash
# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Configure LLM credentials (or select Ollama for local model)
husk init

# Scan a local folder or remote repository link
husk scan https://github.com/example/project.git --detailed

# Generate documentation suite statically or with AI
husk doc . --with-ai

# Query codebase using RAG
husk ask "how does authentication work?" https://github.com/example/project.git
```

---

## 🛠️ CLI Commands

* `husk scan [repo_path_or_url] [--detailed]`: Inventory source files and code symbols.
* `husk graph [repo_path_or_url] [--output path]`: Export visual Mermaid dependency flows.
* `husk hotspots [repo_path_or_url]`: Rank files by complexity × churn risks.
* `husk deadcode [repo_path_or_url]`: List unreferenced files (in-degree = 0).
* `husk init`: Configure LLM credentials (OpenAI, Anthropic, Ollama).
* `husk doc [repo_path_or_url] [--with-ai]`: Generate structured documentation under `/docs`.
* `husk ask "query" [repo_path_or_url]`: Ask natural language questions with git blame context.
