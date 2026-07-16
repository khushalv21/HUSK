import os
import json
import hashlib
from typing import Dict, Optional

class FileCache:
    """
    Manages local filesystem caching for file analysis results using content SHA256 hashes.
    Stores data in `.husk/cache.json`.
    """
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.husk_dir = os.path.join(self.repo_path, ".husk")
        self.cache_file = os.path.join(self.husk_dir, "cache.json")
        self.cache_data: Dict[str, Dict[str, str]] = {}
        self.load()

    def load(self):
        """
        Loads the cache from disk.
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.cache_data = json.load(f)
            except Exception:
                self.cache_data = {}
        else:
            self.cache_data = {}

    def save(self):
        """
        Saves the cache to disk.
        """
        try:
            os.makedirs(self.husk_dir, exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(self.cache_data, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def compute_sha256(file_path: str) -> str:
        """
        Computes the SHA256 hash of a file's content.
        """
        if not os.path.exists(file_path):
            return ""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_summary(self, rel_path: str, file_path: str) -> Optional[str]:
        """
        Retrieves the cached summary for a file if the content hash matches.
        """
        sha = self.compute_sha256(file_path)
        if not sha:
            return None
            
        entry = self.cache_data.get(rel_path)
        if entry and entry.get("sha256") == sha:
            return entry.get("summary")
        return None

    def set_summary(self, rel_path: str, file_path: str, summary: str):
        """
        Updates the cached summary for a file.
        """
        sha = self.compute_sha256(file_path)
        if not sha:
            return
            
        self.cache_data[rel_path] = {
            "sha256": sha,
            "summary": summary
        }
        self.save()

    def get_dir_summary(self, rel_path: str, combined_hash: str) -> Optional[str]:
        """
        Retrieves directory summary if the combined hash matches.
        """
        entry = self.cache_data.get(rel_path)
        if entry and entry.get("sha256") == combined_hash:
            return entry.get("summary")
        return None

    def set_dir_summary(self, rel_path: str, combined_hash: str, summary: str):
        """
        Sets directory summary with combined hash.
        """
        self.cache_data[rel_path] = {
            "sha256": combined_hash,
            "summary": summary
        }
        self.save()

    def clear(self):
        """
        Clears the cache data.
        """
        self.cache_data = {}
        self.save()
