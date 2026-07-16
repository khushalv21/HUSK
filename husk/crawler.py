import os
from typing import Iterator, List, Dict

# Default directories and file patterns to ignore
DEFAULT_IGNORE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
    "target",
    "out",
}

# Mapping of file extensions to tree-sitter language names
EXTENSION_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}

class RepoCrawler:
    """
    Crawls a repository directory to inventory source files and filter by supported languages.
    """
    def __init__(self, root_path: str, ignore_dirs: set = None):
        self.root_path = os.path.abspath(root_path)
        self.ignore_dirs = ignore_dirs if ignore_dirs is not None else DEFAULT_IGNORE_DIRS

    def should_ignore(self, path: str) -> bool:
        """
        Determines if a directory path should be ignored.
        """
        parts = path.replace("\\", "/").split("/")
        for part in parts:
            if part in self.ignore_dirs:
                return True
        return False

    def walk(self) -> Iterator[str]:
        """
        Walks the root path and yields relative paths to non-ignored files.
        """
        for root, dirs, files in os.walk(self.root_path):
            # Prune ignored directories in-place to prevent os.walk from entering them
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            
            # Check if current root itself is ignored (safety check)
            rel_root = os.path.relpath(root, self.root_path)
            if rel_root != "." and self.should_ignore(rel_root):
                continue
                
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.root_path)
                yield rel_path

    def get_inventory(self) -> List[Dict[str, str]]:
        """
        Scans the repository and returns an inventory of supported files with metadata.
        """
        inventory = []
        for rel_path in self.walk():
            ext = os.path.splitext(rel_path)[1].lower()
            if ext in EXTENSION_TO_LANG:
                full_path = os.path.join(self.root_path, rel_path)
                size_bytes = os.path.getsize(full_path)
                inventory.append({
                    "rel_path": rel_path,
                    "language": EXTENSION_TO_LANG[ext],
                    "size_bytes": size_bytes,
                })
        return inventory
