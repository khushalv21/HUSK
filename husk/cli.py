import os
import tempfile
import git
from contextlib import contextmanager
from typing import Optional
import click
from husk.crawler import RepoCrawler
from husk.parser import CodeParser
from husk.graph import DependencyGraphBuilder
from husk.history import GitAnalyzer
from husk.config import ConfigManager
from husk.ai.cache import FileCache
from husk.ai.adapters import get_adapter
from husk.ai.estimator import TokenEstimator
from husk.ai.summarizer import CodeSummarizer
from husk.ai.rag import SyntaxAwareChunker, EmbeddingClient, VectorIndex, QueryEmbeddingCache
from husk.ai.ollama_setup import ensure_models, DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL

def resolve_repo_path(path_or_url: str):
    """
    If path_or_url looks like a Git URL, clones it to a temporary directory
    and returns (temp_path, temp_dir_obj). Otherwise, checks if local path exists and returns (local_path, None).
    """
    is_git_url = path_or_url.startswith(("http://", "https://", "git@", "git://"))
    
    if is_git_url:
        click.echo(f"Cloning remote repository '{path_or_url}'...")
        temp_dir = tempfile.TemporaryDirectory()
        try:
            git.Repo.clone_from(path_or_url, temp_dir.name)
            return temp_dir.name, temp_dir
        except Exception as e:
            try:
                temp_dir.cleanup()
            except Exception:
                pass
            raise click.ClickException(f"Failed to clone remote repository: {e}")
    else:
        abs_path = os.path.abspath(path_or_url)
        if not os.path.exists(abs_path):
            raise click.BadParameter(f"Directory '{path_or_url}' does not exist.")
        if not os.path.isdir(abs_path):
            raise click.BadParameter(f"Path '{path_or_url}' is not a directory.")
        return abs_path, None

def resolve_embedding_backend(provider: str, config_mgr: "ConfigManager", api_key: Optional[str]):
    """
    Determines which provider/key/model to use for embeddings.

    Anthropic has no embeddings API, so RAG indexing/search for anthropic users
    is routed to a local Ollama instance instead of crashing on an unsupported
    provider.
    """
    if provider == "anthropic":
        ollama_url = config_mgr.get("api_url") or "http://localhost:11434"
        return "ollama", ollama_url, "nomic-embed-text", (
            "Anthropic has no embeddings API — using local Ollama "
            f"('{ollama_url}', model 'nomic-embed-text') for search indexing instead."
        )
    if provider == "openai":
        return "openai", api_key, "text-embedding-3-small", None
    if provider in ("ollama", "local"):
        return provider, api_key, "nomic-embed-text", None
    return provider, api_key, None, None

def handle_ai_error(e: Exception, provider: str, model: str) -> str:
    """
    Translates common raw API/network errors into clear, user-friendly error messages.
    """
    msg = str(e)
    # 0. Connection Timeout
    if "timed out" in msg.lower() or "timeout" in msg.lower():
        return (
            f"The request to {provider.upper()} timed out.\n"
            f"  * Hint: Processing large source files on local/CPU LLMs can take more than 5 minutes.\n"
            f"  * Fix: Try running Ollama with GPU acceleration, using a faster quantized model, or setting a budget."
        )
    # 1. Connection refused / Ollama not running
    if "Connection refused" in msg or "Ensure Ollama is running" in msg:
        return (
            f"Could not connect to Ollama at the configured URL.\n"
            f"  * Hint: Please ensure the Ollama app is running locally on your computer."
        )
    # 2. Ollama model not found
    if "model" in msg and "not found" in msg:
        return (
            f"Ollama model '{model}' was not found on your local system.\n"
            f"  * Hint: Run 'ollama pull {model}' in your terminal to download it."
        )
    # 3. OpenAI quota exceeded
    if "insufficient_quota" in msg or "exceeded your current quota" in msg:
        return (
            f"Your {provider.upper()} API key has run out of credit or quota.\n"
            f"  * Hint: Please check your plan and billing details at: https://platform.openai.com/account/billing"
        )
    # 4. Model not found/access denied
    if "model_not_found" in msg or "does not exist" in msg:
        return (
            f"The model '{model}' does not exist or you do not have access to it for {provider.upper()}.\n"
            f"  * Hint: Double-check the model name in your settings or run 'husk init' to update it."
        )
    # 5. Invalid API key
    if "invalid_api_key" in msg or "Incorrect API key" in msg:
        return (
            f"The provided API key for {provider.upper()} is invalid.\n"
            f"  * Hint: Run 'husk init' to set up a new API key."
        )
    # Default fallback
    return f"Failed to communicate with {provider.upper()}: {e}"

MENU_COMMANDS = [
    ("scan [path_or_url] [--detailed] [--with-ai]", "Crawl the repo and inventory files, classes, functions, and imports."),
    ("graph [path_or_url] [--output path]", "Generate and visualize a Mermaid module dependency graph."),
    ("hotspots [path_or_url]", "Rank source files by maintenance risk (Complexity × Git Churn)."),
    ("deadcode [path_or_url]", "Scan for unreferenced files in the module import graph."),
    ("init", "Configure an AI provider — OpenAI, Anthropic, Ollama, or zero-key local."),
    ("doc [path_or_url] [--with-ai]", "Generate structured documentation reports under /docs."),
    ("ask \"query\" [path_or_url] [--rebuild]", "Ask questions about the codebase in plain English via RAG."),
]

def render_menu():
    """
    Renders the colored command menu shared by the bare `husk` invocation and `husk help`.
    """
    click.secho("HUSK", fg="cyan", bold=True)
    click.echo("Your codebase's past, present, and architecture — laid bare.\n")

    click.secho("COMMANDS:", fg="yellow", bold=True)
    for usage, desc in MENU_COMMANDS:
        click.secho(f"  husk {usage}", fg="green", bold=True)
        click.echo(f"    {desc}")
    click.echo("")

    click.secho("EXAMPLES:", fg="yellow", bold=True)
    click.echo("  * Local analysis (no key needed):")
    click.echo("      husk scan . --detailed")
    click.echo("      husk hotspots .")
    click.echo("  * Remote Git scanning:")
    click.echo("      husk scan https://github.com/example/project.git")
    click.echo("  * Zero-API-key AI mode:")
    click.echo("      husk init                 # choose \"local\" — no API key required")
    click.echo("      husk ask \"how does authentication work?\"")
    click.echo("")
    click.echo("Run 'husk COMMAND --help' for full options on any command.")

@click.group(invoke_without_command=True)
@click.version_option("0.2.0", message="Husk v%(version)s")
@click.pass_context
def main(ctx):
    """Husk: A local-first CLI engine for legacy codebases."""
    if ctx.invoked_subcommand is None:
        render_menu()

@main.command()
@click.argument("repo_path", default=".")
@click.option("--detailed", is_flag=True, help="Show extracted classes and functions for each file.")
@click.option("--with-ai", is_flag=True, help="Perform AI summarization of source files.")
@click.option("--ai-budget", type=int, help="Budget limit in cents (e.g. 50 for $0.50).")
@click.option("--dry-run-ai", is_flag=True, help="Estimate AI costs without calling any APIs.")
def scan(repo_path, detailed, with_ai, ai_budget, dry_run_ai):
    """
    Scan the repository and list all supported source files, symbols, and optional AI summaries.
    """
    repo_path, _temp_dir = resolve_repo_path(repo_path)
    click.echo(f"Scanning repository: {os.path.abspath(repo_path)}")
    crawler = RepoCrawler(repo_path)
    inventory = crawler.get_inventory()
    
    if not inventory:
        click.echo("No supported files found (Python, JS, TS).")
        return
        
    click.echo(f"Found {len(inventory)} source files:")
    click.echo("-" * 60)
    
    lang_stats = {}
    total_size = 0
    file_contents = {}
    git_analyzer = GitAnalyzer(repo_path) if detailed else None
    
    for item in inventory:
        rel_path = item["rel_path"]
        lang = item["language"]
        size = item["size_bytes"]
        
        click.echo(f"[{lang.upper()}] {rel_path} ({size} bytes)")
        
        # Accumulate stats
        lang_stats[lang] = lang_stats.get(lang, 0) + 1
        total_size += size
        
        # Read content if we need it for AI
        if with_ai or dry_run_ai:
            try:
                with open(os.path.join(repo_path, rel_path), "r", errors="ignore") as f:
                    file_contents[rel_path] = f.read()
            except Exception:
                file_contents[rel_path] = ""
        
        if detailed:
            # Parse symbols
            try:
                parser = CodeParser(lang)
                result = parser.parse_file(os.path.join(repo_path, rel_path))
                
                if result["classes"]:
                    click.echo("  Classes:")
                    for cls in result["classes"]:
                        blame = git_analyzer.get_line_range_blame(rel_path, cls["start_line"], cls["end_line"])
                        blame_str = f" [Commit {blame['sha']} by {blame['author']}: \"{blame['message']}\"]" if blame["sha"] != "Unknown" else ""
                        click.echo(f"    - {cls['name']} (Lines {cls['start_line']}-{cls['end_line']}){blame_str}")
                if result["functions"]:
                    click.echo("  Functions:")
                    for func in result["functions"]:
                        blame = git_analyzer.get_line_range_blame(rel_path, func["start_line"], func["end_line"])
                        blame_str = f" [Commit {blame['sha']} by {blame['author']}: \"{blame['message']}\"]" if blame["sha"] != "Unknown" else ""
                        click.echo(f"    - {func['name']} (Lines {func['start_line']}-{func['end_line']}){blame_str}")
                if result["imports"]:
                    click.echo(f"  Imports: {', '.join(result['imports'])}")
                click.echo("")
            except Exception as e:
                click.secho(f"  [ERROR] Parsing failed: {e}", fg="red")
                
    click.echo("-" * 60)
    click.echo("Summary Stats:")
    for lang, count in lang_stats.items():
        click.echo(f"  * {lang.capitalize()}: {count} files")
    click.echo(f"  * Total Code Size: {total_size} bytes")
    
    # AI flow
    if with_ai or dry_run_ai:
        config_mgr = ConfigManager()
        provider = config_mgr.get("provider")
        model = config_mgr.get("model")
        
        # Load API keys
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY") or config_mgr.get("api_key")
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY") or config_mgr.get("api_key")
        elif provider in ("ollama", "local"):
            api_key = config_mgr.get("api_url") # Ollama URL acts as host
        else:
            api_key = None
            
        estimator = TokenEstimator(model)
        
        # Calculate tokens for cache misses only
        cache = FileCache(repo_path)
        uncached_tokens = 0
        uncached_files = []
        
        for rel_path, content in file_contents.items():
            full_path = os.path.join(repo_path, rel_path)
            cached_val = cache.get_summary(rel_path, full_path)
            if cached_val is None:
                tokens = estimator.count_tokens(content)
                uncached_tokens += tokens
                uncached_files.append(rel_path)
                
        # Calculate pricing
        est_cost, input_tokens, output_tokens = estimator.calculate_cost(uncached_tokens)
        
        click.echo("-" * 60)
        click.echo("AI Cost Analysis:")
        click.echo(f"  * Provider: {provider.upper()}")
        click.echo(f"  * Model: {model}")
        click.echo(f"  * Total Files to Process: {len(uncached_files)} (out of {len(inventory)} total, {len(inventory) - len(uncached_files)} cached)")
        click.echo(f"  * Estimated Uncached Tokens: {input_tokens} input / {output_tokens} output")
        click.echo(f"  * Estimated Run Cost: {estimator.format_cost(est_cost)}")
        
        if dry_run_ai:
            click.echo("Dry-run only. No API requests will be performed.")
            return
            
        if not api_key and provider != "ollama":
            click.secho(f"\n[ERROR] API key for {provider} not found. Please set the env var or run 'husk init'.", fg="red")
            return
            
        # Check budget limit (budget is in cents, cost is in dollars)
        if ai_budget is not None:
            budget_usd = ai_budget / 100.0
            click.echo(f"  * Budget Limit: {estimator.format_cost(budget_usd)}")
            if est_cost > budget_usd:
                click.secho(f"\n[ABORTED] Estimated cost {estimator.format_cost(est_cost)} exceeds the budget of {estimator.format_cost(budget_usd)}.", fg="red")
                return
                
        if uncached_files:
            click.confirm("\nDo you want to proceed with LLM processing?", abort=True)

            if provider in ("ollama", "local"):
                ensure_models(api_key, [model], log_fn=click.echo, announce_present=False)
            adapter = get_adapter(provider, api_key, model)

            click.echo("\nRunning AI summarization...")
            for rel_path in inventory:
                path = rel_path["rel_path"]
                full_path = os.path.join(repo_path, path)
                summary = cache.get_summary(path, full_path)
                
                if summary:
                    click.echo(f"\n[CACHE HIT] {path}:")
                    click.echo(f"  {summary}")
                else:
                    content = file_contents.get(path, "")
                    if not content.strip():
                        continue
                    click.echo(f"\n[LLM CALL] Summarizing {path}...")
                    
                    try:
                        system_prompt = "You are an expert software archaeologist. Summarize the following source code file in 2-3 concise sentences, explaining its main purpose and key responsibilities."
                        user_prompt = f"File relative path: {path}\nCode content:\n{content}"
                        
                        summary = adapter.generate(user_prompt, system_prompt)
                        cache.set_summary(path, full_path, summary.strip())
                        click.echo(f"  {summary.strip()}")
                    except Exception as e:
                        click.secho(f"  [ERROR] Summarization failed: {e}", fg="red")
        else:
            click.echo("\nAll files are cached. Summary output:")
            for rel_path in inventory:
                path = rel_path["rel_path"]
                full_path = os.path.join(repo_path, path)
                summary = cache.get_summary(path, full_path)
                if summary:
                    click.echo(f"\n* {path}:\n  {summary}")

@main.command()
@click.argument("repo_path", default=".")
@click.option("--output", "-o", type=click.Path(), help="Path to write the Mermaid output file.")
def graph(repo_path, output):
    """
    Generate and display the module dependency graph for the repository.
    """
    repo_path, _temp_dir = resolve_repo_path(repo_path)
    click.echo(f"Analyzing dependencies in: {os.path.abspath(repo_path)}")
    crawler = RepoCrawler(repo_path)
    inventory = crawler.get_inventory()
    
    parsed_data = {}
    for item in inventory:
        rel_path = item["rel_path"]
        lang = item["language"]
        
        try:
            parser = CodeParser(lang)
            parsed_data[rel_path] = parser.parse_file(os.path.join(repo_path, rel_path))
        except Exception as e:
            click.echo(f"Warning: Failed to parse {rel_path}: {e}", err=True)
            parsed_data[rel_path] = {"classes": [], "functions": [], "imports": []}
            
    builder = DependencyGraphBuilder(inventory, parsed_data)
    builder.build()
    mermaid_str = builder.to_mermaid()
    
    click.echo("\n--- Mermaid Graph ---")
    click.echo(mermaid_str)
    click.echo("----------------------\n")
    
    if output:
        # Create output directories if needed
        out_dir = os.path.dirname(output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        with open(output, "w") as f:
            f.write("# Module Dependency Graph\n\n```mermaid\n")
            f.write(mermaid_str)
            f.write("\n```\n")
        click.secho(f"Wrote dependency graph to {output}", fg="green")

@main.command()
@click.argument("repo_path", default=".")
def hotspots(repo_path):
    """
    Identify complexity + churn hotspots in the repository.
    """
    repo_path, _temp_dir = resolve_repo_path(repo_path)
    click.echo(f"Calculating hotspots in: {os.path.abspath(repo_path)}")
    crawler = RepoCrawler(repo_path)
    inventory = crawler.get_inventory()
    git_analyzer = GitAnalyzer(repo_path)
    
    if not git_analyzer.is_git_repo():
        click.echo("Warning: Not a git repository. Git churn statistics will not be available.", err=True)

    hotspots_list = []
    all_rel_paths = [item["rel_path"] for item in inventory]
    bulk_metrics = git_analyzer.get_bulk_file_metrics(all_rel_paths)

    for item in inventory:
        rel_path = item["rel_path"]
        lang = item["language"]

        # Calculate complexity
        try:
            parser = CodeParser(lang)
            complexity = parser.calculate_complexity(os.path.join(repo_path, rel_path))
        except Exception:
            complexity = 1

        # Get churn
        git_metrics = bulk_metrics[rel_path]
        churn = git_metrics["churn"]

        score = complexity * churn
        hotspots_list.append({
            "rel_path": rel_path,
            "complexity": complexity,
            "churn": churn,
            "authors": git_metrics["authors"],
            "last_modified": git_metrics["last_modified"],
            "score": score
        })
        
    # Sort by score descending, then by churn descending, then complexity descending
    hotspots_list.sort(key=lambda x: (x["score"], x["churn"], x["complexity"]), reverse=True)
    
    click.echo("\n--- Code Hotspot Report (ranked by Complexity * Churn) ---")
    click.echo(f"{'File Path':<40} | {'Complexity':<10} | {'Churn':<6} | {'Score':<8} | {'Authors':<7} | {'Last Modified':<19}")
    click.echo("-" * 102)
    
    for h in hotspots_list:
        click.echo(f"{h['rel_path']:<40} | {h['complexity']:<10} | {h['churn']:<6} | {h['score']:<8} | {h['authors']:<7} | {h['last_modified']:<19}")

@main.command()
@click.argument("repo_path", default=".")
def deadcode(repo_path):
    """
    Identify potential dead or unreferenced source files.
    """
    repo_path, _temp_dir = resolve_repo_path(repo_path)
    click.echo(f"Scanning for dead code in: {os.path.abspath(repo_path)}")
    crawler = RepoCrawler(repo_path)
    inventory = crawler.get_inventory()
    
    parsed_data = {}
    for item in inventory:
        rel_path = item["rel_path"]
        lang = item["language"]
        try:
            parser = CodeParser(lang)
            parsed_data[rel_path] = parser.parse_file(os.path.join(repo_path, rel_path))
        except Exception:
            parsed_data[rel_path] = {"classes": [], "functions": [], "imports": []}
            
    builder = DependencyGraphBuilder(inventory, parsed_data)
    graph = builder.build()
    
    # Dead code candidates have in-degree of 0 (no imports reference them)
    dead_candidates = []
    for node in graph.nodes:
        if graph.in_degree(node) == 0:
            # Check if this node is a typical entrypoint (e.g. cli.py, main.py, index.js, test files)
            base = os.path.basename(node).lower()
            is_entrypoint = base in ("main.py", "cli.py", "index.js", "app.py", "setup.py", "wsgi.py") or base.startswith("test_")
            dead_candidates.append({
                "rel_path": node,
                "is_entrypoint": is_entrypoint
            })
            
    if not dead_candidates:
        click.echo("No unreferenced files found!")
        return
        
    click.echo("\n--- Unreferenced File Candidates (In-Degree = 0) ---")
    click.echo(f"{'File Path':<50} | {'Status':<25}")
    click.echo("-" * 80)
    for c in dead_candidates:
        status = "Potential Entry Point (Expected)" if c["is_entrypoint"] else "Unreferenced (Likely Dead Code)"
        click.echo(f"{c['rel_path']:<50} | {status:<25}")


@main.command()
def init():
    """
    Initialize and configure the Husk settings file (~/.husk/config.yaml).
    """
    config_mgr = ConfigManager()

    click.echo("--- Husk Config Wizard ---")
    click.echo("Providers: openai, anthropic, ollama (bring your own local model), "
                "or local (no key needed — auto-installs a small default model via Ollama).\n")

    provider = click.prompt(
        "Choose an AI provider",
        type=click.Choice(["openai", "anthropic", "ollama", "local"], case_sensitive=False),
        default=config_mgr.get("provider")
    ).lower()

    default_model = "gpt-4o-mini"
    if provider == "anthropic":
        default_model = "claude-3-5-sonnet-20241022"
    elif provider == "ollama":
        default_model = "llama3"
    elif provider == "local":
        default_model = DEFAULT_CHAT_MODEL

    if provider == "local":
        # Zero-key path: skip the model prompt entirely and use the curated small default.
        model = default_model
        click.echo(f"Using default local model: {model} (no API key required).")
    else:
        model = click.prompt(
            f"Enter the model name to use",
            type=str,
            default=config_mgr.get("model") if config_mgr.get("provider") == provider else default_model
        )

    api_key = ""
    api_url = "http://localhost:11434"

    if provider in ("openai", "anthropic"):
        api_key = click.prompt(
            "Enter your API key (leave empty to use env variables)",
            default="",
            show_default=False,
            hide_input=True
        )
    elif provider in ("ollama", "local"):
        api_url = click.prompt(
            "Enter Ollama server Host URL",
            default=config_mgr.get("api_url") or "http://localhost:11434"
        )

    config_mgr.set("provider", provider)
    config_mgr.set("model", model)
    if api_key:
        config_mgr.set("api_key", api_key)
    if provider in ("ollama", "local"):
        config_mgr.set("api_url", api_url)

    click.secho(f"\nConfiguration saved to {config_mgr.config_path}", fg="green")

    if provider == "local":
        click.echo("\nProvisioning local models via Ollama (no key needed)...")
        ok = ensure_models(api_url, [model, DEFAULT_EMBED_MODEL], log_fn=click.echo)
        if ok:
            click.echo("\nHusk is ready to use with no API key — try 'husk ask \"...\"' or 'husk doc --with-ai'.")
        else:
            click.echo(
                "\n[WARNING] Could not finish provisioning automatically. "
                "Make sure Ollama is installed and running, then re-run 'husk init' "
                "or manually run: ollama pull " + model
            )

@main.command()
@click.argument("repo_path", default=".")
@click.option("--with-ai", is_flag=True, help="Perform AI map-reduce summarization.")
@click.option("--ai-budget", type=int, help="Budget limit in cents (e.g. 50 for $0.50).")
@click.option("--dry-run-ai", is_flag=True, help="Estimate AI costs without calling any APIs.")
def doc(repo_path, with_ai, ai_budget, dry_run_ai):
    """
    Generate hierarchical summaries (files, modules) and system-level docs.
    """
    repo_path, _temp_dir = resolve_repo_path(repo_path)
    click.echo(f"Initializing documentation pipeline in: {os.path.abspath(repo_path)}")
    crawler = RepoCrawler(repo_path)
    inventory = crawler.get_inventory()
    
    if not inventory:
        click.echo("No source files found.")
        return
        
    config_mgr = ConfigManager()
    provider = config_mgr.get("provider")
    model = config_mgr.get("model")
    
    # Load API keys
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY") or config_mgr.get("api_key")
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY") or config_mgr.get("api_key")
    elif provider in ("ollama", "local"):
        api_key = config_mgr.get("api_url")
    else:
        api_key = None
        
    cache = FileCache(repo_path)
    estimator = TokenEstimator(model)
    
    # We will build a dummy adapter for token counting so we don't need keys for dry run
    adapter = get_adapter(provider, api_key or "dummy", model)
    summarizer = CodeSummarizer(repo_path, adapter, cache)
    
    # Group files by parent directory
    dir_groups = summarizer.group_files_by_directory(inventory)
    
    # Gather file contents
    file_contents = {}
    for item in inventory:
        path = item["rel_path"]
        try:
            with open(os.path.join(repo_path, path), "r", errors="ignore") as f:
                file_contents[path] = f.read()
        except Exception:
            file_contents[path] = ""

    # Estimate hierarchical cost:
    # 1. Map step (Files)
    map_tokens = 0
    uncached_files = []
    for path, content in file_contents.items():
        full_path = os.path.join(repo_path, path)
        if cache.get_summary(path, full_path) is None:
            map_tokens += estimator.count_tokens(content)
            uncached_files.append(path)
            
    # 2. Reduce 1 step (Modules)
    reduce1_tokens = 0
    uncached_dirs = []
    for rdir, files in dir_groups.items():
        combined_hash = summarizer.get_directory_combined_hash(rdir, files)
        if cache.get_dir_summary(rdir, combined_hash) is None:
            # Approx: each child summary is ~150 tokens in the prompt
            reduce1_tokens += len(files) * 150
            uncached_dirs.append(rdir)
            
    # 3. Reduce 2 step (System Architecture)
    reduce2_tokens = 0
    arch_file = os.path.join(repo_path, "docs", "architecture.md")
    if not os.path.exists(arch_file) or uncached_dirs:
        reduce2_tokens += len(dir_groups) * 150
        
    total_input_tokens = map_tokens + reduce1_tokens + reduce2_tokens
    # Output tokens: 1 summary per uncached file (~150 tokens) + 1 per uncached dir (~150 tokens) + architecture (~800 tokens)
    total_output_tokens = (len(uncached_files) * 150) + (len(uncached_dirs) * 150)
    if reduce2_tokens > 0:
        total_output_tokens += 800
        
    # Cost calculation
    pricing = estimator.pricing
    input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
    output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
    est_cost = input_cost + output_cost
    
    click.echo("-" * 60)
    click.echo("Map-Reduce Summarization Cost Estimate:")
    click.echo(f"  * Provider: {provider.upper()}")
    click.echo(f"  * Model: {model}")
    click.echo(f"  * Uncached Files (Map): {len(uncached_files)} / {len(inventory)}")
    click.echo(f"  * Uncached Modules (Reduce 1): {len(uncached_dirs)} / {len(dir_groups)}")
    click.echo(f"  * System Doc Regeneration (Reduce 2): {'Yes' if reduce2_tokens > 0 else 'No'}")
    click.echo(f"  * Estimated Total Input Tokens: {total_input_tokens}")
    click.echo(f"  * Estimated Total Output Tokens: {total_output_tokens}")
    click.echo(f"  * Estimated Total Cost: {estimator.format_cost(est_cost)}")
    click.echo("-" * 60)
    
    if dry_run_ai:
        click.echo("Dry-run complete. No API calls made.")
        return
        
    if not with_ai:
        click.echo("AI generation not requested. Running static documentation suite...")
    else:
        if not api_key and provider != "ollama":
            click.secho(f"\n[ERROR] API key for {provider} not found. Please set the env var or run 'husk init'.", fg="red")
            return
            
        # Check budget
        if ai_budget is not None:
            budget_usd = ai_budget / 100.0
            click.echo(f"  * Budget Limit: {estimator.format_cost(budget_usd)}")
            if est_cost > budget_usd:
                click.secho(f"\n[ABORTED] Estimated cost {estimator.format_cost(est_cost)} exceeds the budget of {estimator.format_cost(budget_usd)}.", fg="red")
                return
                
        # Confirm
        click.confirm("\nProceed with Map-Reduce AI documentation generation?", abort=True)
        
        try:
            if provider in ("ollama", "local"):
                ensure_models(api_key, [model], log_fn=click.echo, announce_present=False)
            # Initialize real adapter
            real_adapter = get_adapter(provider, api_key, model)
            summarizer.adapter = real_adapter
            
            # 1. Map step: Summarize all files
            click.echo("\n--- Step 1: Generating File Summaries (Map) ---")
            file_summaries = {}
            for path in sorted(file_contents.keys()):
                content = file_contents[path]
                if not content.strip():
                    continue
                summary = summarizer.summarize_file(path, content, log_fn=click.echo)
                file_summaries[path] = summary
                
            # 2. Reduce 1 step: Summarize modules/directories
            click.echo("\n--- Step 2: Generating Module Rollups (Reduce 1) ---")
            module_summaries = {}
            for rdir, files in sorted(dir_groups.items()):
                summary = summarizer.summarize_directory(rdir, files, file_summaries, log_fn=click.echo)
                module_summaries[rdir] = summary
                click.echo(f"  Module '{rdir}': {summary}\n")
                
            # 3. Reduce 2 step: Generate system architecture overview
            click.echo("\n--- Step 3: Generating System Architecture (Reduce 2) ---")
            click.echo("[LLM CALL] Generating docs/architecture.md...")
            arch_markdown = summarizer.generate_system_architecture(module_summaries)
            
            # Write to docs/architecture.md
            docs_dir = os.path.join(repo_path, "docs")
            os.makedirs(docs_dir, exist_ok=True)
            with open(arch_file, "w") as f:
                f.write(arch_markdown)
            click.secho(f"Wrote system architecture document to {arch_file}", fg="green")
        except Exception as e:
            raise click.ClickException(handle_ai_error(e, provider, model))

    # --- ALWAYS Write Static Documentation Files ---
    click.echo("\nWriting static documentation files to docs/...")
    docs_dir = os.path.join(repo_path, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    graph = None
    
    # 1. dependency_graph.md
    graph_file = os.path.join(docs_dir, "dependency_graph.md")
    try:
        parsed_data = {}
        for item in inventory:
            path = item["rel_path"]
            lang = item["language"]
            try:
                parser = CodeParser(lang)
                parsed_data[path] = parser.parse_file(os.path.join(repo_path, path))
            except Exception:
                parsed_data[path] = {"classes": [], "functions": [], "imports": []}
                
        graph_builder = DependencyGraphBuilder(inventory, parsed_data)
        graph = graph_builder.build()
        mermaid_str = graph_builder.to_mermaid()
        with open(graph_file, "w") as f:
            f.write(f"# Dependency Graph\n\n```mermaid\n{mermaid_str}\n```\n")
        click.secho(f"  * Wrote {graph_file}", fg="green")
    except Exception as e:
        click.secho(f"  [Warning] Failed to generate dependency graph: {e}", fg="yellow")
        
    # 2. hotspots.md
    hotspots_file = os.path.join(docs_dir, "hotspots.md")
    try:
        git_analyzer = GitAnalyzer(repo_path)
        hotspots_list = []
        bulk_metrics = git_analyzer.get_bulk_file_metrics([item["rel_path"] for item in inventory])
        for item in inventory:
            path = item["rel_path"]
            lang = item["language"]
            try:
                parser = CodeParser(lang)
                complexity = parser.calculate_complexity(os.path.join(repo_path, path))
            except Exception:
                complexity = 1
            metrics = bulk_metrics[path]
            score = complexity * metrics["churn"]
            hotspots_list.append({
                "path": path,
                "complexity": complexity,
                "churn": metrics["churn"],
                "score": score,
                "authors": metrics["authors"],
                "last_modified": metrics["last_modified"]
            })
        hotspots_list.sort(key=lambda x: x["score"], reverse=True)
        with open(hotspots_file, "w") as f:
            f.write("# Hotspots Analysis\n\nRankings based on complexity × churn:\n\n")
            f.write("| File | Complexity | Churn | Hotspot Score | Unique Authors | Last Modified |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            for h in hotspots_list:
                f.write(f"| {h['path']} | {h['complexity']} | {h['churn']} | {h['score']} | {h['authors']} | {h['last_modified']} |\n")
        click.secho(f"  * Wrote {hotspots_file}", fg="green")
    except Exception as e:
        click.secho(f"  [Warning] Failed to generate hotspots report: {e}", fg="yellow")
        
    # 3. deadcode.md
    deadcode_file = os.path.join(docs_dir, "deadcode.md")
    try:
        dead_candidates = []
        if graph is not None:
            for node in graph.nodes:
                if graph.in_degree(node) == 0:
                    dead_candidates.append(node)
            with open(deadcode_file, "w") as f:
                f.write("# Dead Code Detection\n\nPotential dead or unreferenced source files (in-degree = 0):\n\n")
                if dead_candidates:
                    for c in sorted(dead_candidates):
                        f.write(f"- `{c}`\n")
                else:
                    f.write("No dead code candidates detected.\n")
            click.secho(f"  * Wrote {deadcode_file}", fg="green")
        else:
            click.secho("  [Warning] Skipping dead code report: dependency graph was not built.", fg="yellow")
    except Exception as e:
        click.secho(f"  [Warning] Failed to generate dead code report: {e}", fg="yellow")
        
    # 4. index.md
    index_file = os.path.join(docs_dir, "index.md")
    try:
        summaries_rows = []
        for item in inventory:
            path = item["rel_path"]
            full_path = os.path.join(repo_path, path)
            summary = cache.get_summary(path, full_path) or "*(No summary generated)*"
            summaries_rows.append(f"| `{path}` | {summary} |")
        summaries_table = "\n".join(summaries_rows)
        with open(index_file, "w") as f:
            f.write("# Codebase Documentation Index\n\n")
            f.write("Welcome to the documentation suite generated by **Husk**.\n\n")
            f.write("## 📑 Document Inventory\n")
            if os.path.exists(os.path.join(docs_dir, "architecture.md")):
                f.write("- 📐 **[System Architecture](architecture.md):** High-level component structures and design patterns.\n")
            else:
                f.write("- 📐 **System Architecture:** *(Not generated. Run `husk doc --with-ai` to generate)*\n")
            f.write("- 🔗 **[Dependency Graph](dependency_graph.md):** Module interaction flowchart.\n")
            f.write("- ⚡ **[Hotspots Analysis](hotspots.md):** Churn & complexity indicators.\n")
            f.write("- 🔍 **[Dead Code Detection](deadcode.md):** Unreferenced components report.\n\n")
            f.write("## 🗄️ File Summaries\n\n")
            f.write("| File | Summary |\n")
            f.write("| --- | --- |\n")
            f.write(summaries_table + "\n")
        click.secho(f"  * Wrote {index_file}", fg="green")
    except Exception as e:
        click.secho(f"  [Warning] Failed to generate index.md: {e}", fg="yellow")
        
    click.secho("\nDocumentation suite successfully updated!", fg="green")

def _print_ask_results(query, search_results, adapter, provider, model):
    """
    Runs one query against an already-searched result set: prints scored citations
    with snippets, then synthesizes and prints the LLM answer. Shared by single-shot
    and interactive `husk ask` so the two modes render identically.
    """
    if not search_results:
        click.echo("No relevant code chunks found.")
        return

    context_blocks = []
    citations = []
    for result in search_results:
        chunk = result["chunk"]
        meta = chunk["metadata"]
        pct = round(result["score"] * 100)
        cite = f"{meta['rel_path']} (Lines {meta['start_line']}-{meta['end_line']}, Type: {meta['type']})"
        citations.append((cite, pct, result["vector_score"], result["keyword_score"], chunk["text"]))

        block = f"--- Citation: {cite} (Relevance: {pct}%) ---\n"
        if "why" in meta:
            block += f"// Why annotation: {meta['why']}\n"
        block += chunk["text"] + "\n"
        context_blocks.append(block)

    context_text = "\n".join(context_blocks)

    click.echo("Synthesizing answer...")
    system_prompt = (
        "You are an expert software archaeologist. Answer the user's question about the codebase "
        "using the provided relevant code chunks and documentation snippets. Provide structured, "
        "accurate explanations. Cite file paths and lines wherever appropriate."
    )
    user_prompt = f"Relevant Codebase Context:\n\n{context_text}\n\nQuestion: {query}"

    try:
        answer = adapter.generate(user_prompt, system_prompt)
    except Exception as e:
        raise click.ClickException(handle_ai_error(e, provider, model))

    click.secho("\n--- Answer ---", fg="cyan", bold=True)
    click.echo(answer)
    click.secho("\n--- Sources & Citations ---", fg="cyan", bold=True)
    for cite, pct, vector_score, keyword_score, text in citations:
        click.secho(f"- {cite}  [{pct}% match — vector {vector_score:.2f}, keyword {keyword_score:.2f}]", fg="green")
        snippet_lines = text.splitlines()[:4]
        for line in snippet_lines:
            click.secho(f"    {line}", fg="bright_black")
        if len(text.splitlines()) > 4:
            click.secho("    ...", fg="bright_black")

@main.command()
@click.argument("query", required=False, default=None)
@click.argument("repo_path", default=".")
@click.option("--rebuild", is_flag=True, help="Force rebuild the RAG search index.")
@click.option("--interactive", "-i", is_flag=True, help="Start an interactive Q&A session instead of a single query.")
def ask(query, repo_path, rebuild, interactive):
    """
    Query the codebase in plain English using RAG search.
    """
    if not interactive and not query:
        raise click.UsageError("QUERY is required unless --interactive/-i is set.")
    repo_path, _temp_dir = resolve_repo_path(repo_path)
    config_mgr = ConfigManager()
    provider = config_mgr.get("provider")
    model = config_mgr.get("model")
    
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY") or config_mgr.get("api_key")
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY") or config_mgr.get("api_key")
    elif provider in ("ollama", "local"):
        api_key = config_mgr.get("api_url")
    else:
        api_key = None
        
    if not api_key and provider != "ollama":
        click.secho(f"[ERROR] API key for {provider} not found. Please set the env var or run 'husk init'.", fg="red")
        return

    index_file = os.path.join(repo_path, ".husk", "rag_index.json")
    index = VectorIndex(index_file)

    emb_provider, emb_api_key, emb_model, emb_note = resolve_embedding_backend(provider, config_mgr, api_key)
    if emb_note:
        click.echo(emb_note)
    if emb_model is None:
        raise click.ClickException(
            f"No embedding backend available for provider '{provider}'. "
            "Configure 'openai' or 'ollama' (or run 'husk init') to use 'husk ask'."
        )

    # Pre-flight: if we're using Ollama for embeddings (directly, or as the
    # Anthropic-has-no-embeddings fallback), make sure the model is actually pulled
    # instead of failing deep inside the indexing loop.
    if emb_provider in ("ollama", "local"):
        ensure_models(emb_api_key, [emb_model], log_fn=click.echo, announce_present=False)

    emb_client = EmbeddingClient(emb_provider, emb_api_key, emb_model)

    if rebuild:
        click.echo("--rebuild passed: wiping search index for a full re-index.")
        index.clear()

    crawler = RepoCrawler(repo_path)
    inventory = crawler.get_inventory()
    cache = FileCache(repo_path)
    git_analyzer = GitAnalyzer(repo_path)

    known_paths = {item["rel_path"] for item in inventory}
    removed = index.remove_stale_files(known_paths)
    if removed:
        click.echo(f"Removed {len(removed)} deleted file(s) from index.")

    # Only files that are new or whose content hash changed since the last index need
    # re-chunking/re-embedding — unchanged files are skipped entirely.
    changed_items = []
    for item in inventory:
        path = item["rel_path"]
        full_path = os.path.join(repo_path, path)
        sha = FileCache.compute_sha256(full_path)
        if sha and index.get_file_hash(path) != sha:
            changed_items.append((item, sha))

    if changed_items:
        click.echo(f"Indexing {len(changed_items)} new/changed file(s)... (this may take a moment)")
        for item, sha in changed_items:
            path = item["rel_path"]
            lang = item["language"]
            full_path = os.path.join(repo_path, path)

            try:
                with open(full_path, "r", errors="ignore") as f:
                    content = f.read()

                parser = CodeParser(lang)
                parsed_data = parser.parse_file(full_path)
                file_chunks = SyntaxAwareChunker.chunk_file(path, content, parsed_data)

                new_chunk_objs = []
                for chunk in file_chunks:
                    meta = chunk["metadata"]
                    # Add git blame why annotation
                    blame = git_analyzer.get_line_range_blame(path, meta["start_line"], meta["end_line"])
                    if blame["sha"] != "Unknown":
                        meta["why"] = f"Last changed in commit {blame['sha']} by {blame['author']}: '{blame['message']}'"

                    emb = emb_client.get_embedding(chunk["text"])
                    new_chunk_objs.append({"text": chunk["text"], "metadata": meta, "embedding": emb})

                # Include the file's AI summary (if any) as an extra searchable chunk
                summary = cache.get_summary(path, full_path)
                if summary:
                    meta = {
                        "rel_path": path,
                        "type": "file_summary",
                        "name": "summary",
                        "start_line": 1,
                        "end_line": 1
                    }
                    text = f"// File summary for {path}\n{summary}"
                    emb = emb_client.get_embedding(text)
                    new_chunk_objs.append({"text": text, "metadata": meta, "embedding": emb})

                index.update_file(path, sha, new_chunk_objs)
            except Exception as e:
                click.echo(f"Warning: Failed to index file {path}: {e}")

        index.save()
        click.echo(f"Index now has {len(index.chunks)} chunks across {len(index.file_hashes)} file(s).")
    else:
        click.echo("Index is up to date. No files changed since the last index.")

    # Chat model pre-flight + adapter, built once and reused across queries (important for
    # --interactive, where re-pulling/reconnecting per question would be wasteful).
    if provider in ("ollama", "local"):
        ensure_models(api_key, [model], log_fn=click.echo, announce_present=False)
    adapter = get_adapter(provider, api_key, model)
    query_cache = QueryEmbeddingCache(repo_path)

    def run_query(q: str):
        click.echo(f"Searching index for: '{q}'...")
        try:
            query_emb = query_cache.get(emb_model, q)
            if query_emb is None:
                query_emb = emb_client.get_embedding(q)
                query_cache.set(emb_model, q, query_emb)
            search_results = index.hybrid_search(query_emb, q, top_k=5)
        except Exception as e:
            raise click.ClickException(handle_ai_error(e, emb_provider, emb_model))

        _print_ask_results(q, search_results, adapter, provider, model)

    if interactive:
        click.secho(
            "\nInteractive mode — ask as many questions as you like. Type 'exit', 'quit', or Ctrl-D to stop.",
            fg="cyan", bold=True
        )
        while True:
            try:
                q = click.prompt("\nAsk", prompt_suffix="> ")
            except (EOFError, click.exceptions.Abort):
                click.echo("\nExiting interactive mode.")
                break
            q = q.strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                click.echo("Exiting interactive mode.")
                break
            run_query(q)
    else:
        run_query(query)

@main.command()
def help():
    """
    Display detailed help menu and usage examples for all Husk commands.
    """
    render_menu()

if __name__ == "__main__":
    main()
