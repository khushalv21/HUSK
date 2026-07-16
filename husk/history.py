import os
from typing import Dict, Set
from git import Repo

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

    def get_file_metrics(self, rel_path: str) -> Dict[str, any]:
        """
        Gets git metrics for a specific file.
        """
        metrics = {
            "churn": 0,
            "authors": 0,
            "last_modified": "Unknown",
        }
        
        if not self.repo:
            return metrics
            
        try:
            # Get all commits that modified this file
            commits = list(self.repo.iter_commits(paths=rel_path))
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

