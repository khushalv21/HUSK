import os
from typing import Dict, List, Set
from git import Repo

# Caps how many commits `get_file_metrics` will walk per file. Without a cap,
# `iter_commits(paths=...)` walks a file's ENTIRE history on every call, so on a large,
# long-lived repo (e.g. Django's real history), running hotspots/doc across a few thousand
# files becomes minutes-to-hours of per-file `git log` subprocess calls. Capping keeps churn
# a meaningful *relative* ranking signal (files changed >= this many times all rank as
# equally "very churny", which is fine for a hotspot ranking) while keeping the command fast.
DEFAULT_CHURN_MAX_COMMITS = 200

class GitAnalyzer:
    """
    Analyzes git history to determine file churn, author counts, and metadata.
    """
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        try:
            self.repo = Repo(self.repo_path, search_parent_directories=True)
        except Exception:
            self.repo = None

    def is_git_repo(self) -> bool:
        """
        Returns True if the path is inside a valid git repository.
        """
        return self.repo is not None

    def get_file_metrics(self, rel_path: str, max_commits: int = DEFAULT_CHURN_MAX_COMMITS) -> Dict[str, any]:
        """
        Gets git metrics for a specific file, capping the commit walk at `max_commits`
        (see DEFAULT_CHURN_MAX_COMMITS) so this stays fast on large, long-lived repos.
        """
        metrics = {
            "churn": 0,
            "authors": 0,
            "last_modified": "Unknown",
        }

        if not self.repo:
            return metrics

        try:
            # Get the most recent commits that modified this file, up to max_commits
            commits = list(self.repo.iter_commits(paths=rel_path, max_count=max_commits))
            metrics["churn"] = len(commits)

            if commits:
                authors: Set[str] = set()
                for c in commits:
                    if c.author:
                        author_id = c.author.email or c.author.name
                        if author_id:
                            authors.add(author_id)
                metrics["authors"] = len(authors)
                metrics["last_modified"] = commits[0].committed_datetime.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

        return metrics

    def get_bulk_file_metrics(
        self, rel_paths: List[str], max_commits: int = 5000
    ) -> Dict[str, Dict[str, any]]:
        """
        Computes churn/authors/last_modified for many files at once from a single
        `git log` walk, instead of spawning one `git` subprocess per file.

        `get_file_metrics` spawns a fresh `git rev-list`/`git log` process per file, which
        is fine for a handful of files but dominates runtime on repos with thousands of
        files (measured: ~2 minutes for ~3,000 files on Django, almost entirely process-spawn
        overhead rather than history-walk depth). This walks the last `max_commits` commits
        of the whole repo exactly once and aggregates per-file stats from that single pass.
        """
        empty = lambda: {"churn": 0, "authors": 0, "last_modified": "Unknown"}
        results = {p: empty() for p in rel_paths}

        if not self.repo:
            return results

        known_paths = set(rel_paths)
        authors_by_path: Dict[str, Set[str]] = {p: set() for p in rel_paths}

        try:
            log_output = self.repo.git.log(
                f"-n{max_commits}", "--name-only", "--pretty=format:__COMMIT__%H|%ae|%an|%cI"
            )
        except Exception:
            return results

        current_email = current_name = current_date = None
        for line in log_output.splitlines():
            if line.startswith("__COMMIT__"):
                header = line[len("__COMMIT__"):]
                parts = header.split("|", 3)
                if len(parts) == 4:
                    _sha, current_email, current_name, current_date = parts
                else:
                    current_email = current_name = current_date = None
                continue

            path = line.strip()
            if not path or path not in known_paths:
                continue

            entry = results[path]
            entry["churn"] += 1
            author_id = current_email or current_name
            if author_id:
                authors_by_path[path].add(author_id)
            # git log is newest-first, so the first hit for a path is its most recent change.
            if entry["last_modified"] == "Unknown" and current_date:
                entry["last_modified"] = current_date[:19].replace("T", " ")

        for p, entry in results.items():
            entry["authors"] = len(authors_by_path[p])

        return results

    def get_line_range_blame(self, rel_path: str, start_line: int, end_line: int) -> dict:
        """
        Runs git blame on a specific line range and returns the details of the latest commit.
        """
        metrics = {
            "sha": "Unknown",
            "author": "Unknown",
            "message": "No commit info available",
            "date": "Unknown"
        }
        
        if not self.repo:
            return metrics
            
        try:
            blame_output = self.repo.git.blame("-L", f"{start_line},{end_line}", "--", rel_path)
            shas = set()
            for line in blame_output.splitlines():
                if line.strip():
                    parts = line.split()
                    if parts:
                        sha = parts[0]
                        # Strip boundary carat
                        if sha.startswith("^"):
                            sha = sha[1:]
                        # Verify it's a valid hex hash prefix
                        if all(c in "0123456789abcdefABCDEF" for c in sha) and len(sha) >= 7:
                            shas.add(sha)
            
            latest_commit = None
            for sha in shas:
                try:
                    commit = self.repo.commit(sha)
                    if latest_commit is None or commit.committed_date > latest_commit.committed_date:
                        latest_commit = commit
                except Exception:
                    pass
                    
            if latest_commit:
                metrics["sha"] = latest_commit.hexsha[:8]
                metrics["author"] = latest_commit.author.name
                metrics["message"] = latest_commit.message.splitlines()[0] if latest_commit.message else ""
                metrics["date"] = latest_commit.committed_datetime.strftime("%Y-%m-%d")
        except Exception:
            pass
            
        return metrics

