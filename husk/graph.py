import os
import networkx as nx
from typing import List, Dict, Set, Tuple

class DependencyGraphBuilder:
    """
    Builds and exports dependency graphs representing connections between workspace modules.
    """
    def __init__(self, inventory: List[Dict[str, str]], parsed_data: Dict[str, Dict]):
        self.inventory = inventory
        # Map of rel_path -> parsed info {"classes": [], "functions": [], "imports": []}
        self.parsed_data = parsed_data
        self.graph = nx.DiGraph()

        # Add all files as nodes
        for item in inventory:
            self.graph.add_node(item["rel_path"], language=item["language"])

        # Precompute each Python file's dotted module name once, instead of recomputing
        # os.path.splitext/replace on every (importing_file, import) pair inside
        # resolve_python_import's inner loop — on a repo with thousands of files and
        # imports, that redundant string work dominates `husk graph`/`husk deadcode` runtime.
        self._python_file_mods: List[Tuple[str, str]] = [
            (os.path.splitext(item["rel_path"])[0].replace("/", "."), item["rel_path"])
            for item in inventory
            if item["language"] == "python"
        ]

    def resolve_python_import(self, importing_file: str, import_name: str) -> str:
        """
        Attempts to resolve a Python module import string to a relative workspace path.
        """
        # Check files in our inventory
        for file_mod, file_rel in self._python_file_mods:
            # Match direct module or package init
            if file_mod == import_name or file_mod.endswith("." + import_name):
                return file_rel

            # Match sub-modules if importing parent package
            if import_name.startswith(file_mod) and file_mod != "":
                return file_rel

        return ""

    def resolve_js_ts_import(self, importing_file: str, import_path: str) -> str:
        """
        Attempts to resolve JS/TS relative imports to a relative workspace path.
        """
        if not (import_path.startswith("./") or import_path.startswith("../")):
            # Probably a third-party npm package, ignore for workspace graph
            return ""
            
        importing_dir = os.path.dirname(importing_file)
        # Compute potential path
        target_path = os.path.normpath(os.path.join(importing_dir, import_path))
        
        # Extensions to try
        extensions = [".js", ".jsx", ".ts", ".tsx", "/index.js", "/index.jsx", "/index.ts", "/index.tsx"]
        
        for ext in extensions:
            test_path = target_path + ext if not ext.startswith("/index") else target_path + ext
            # Check if this file exists in our inventory
            for item in self.inventory:
                if os.path.normpath(item["rel_path"]) == os.path.normpath(test_path):
                    return item["rel_path"]
                    
        return ""

    def resolve_java_import(self, importing_file: str, import_name: str) -> str:
        """
        Attempts to resolve a Java fully-qualified import (e.g. "com.foo.Bar") to a
        relative workspace path (e.g. "src/main/java/com/foo/Bar.java").
        """
        # Wildcard/package imports (e.g. "com.foo.*") reference a whole package, not a
        # single file, so there's nothing specific to resolve to.
        if import_name.endswith(".*"):
            return ""

        candidate_suffix = "/".join(import_name.split(".")) + ".java"

        for item in self.inventory:
            if item["language"] != "java":
                continue

            file_rel = item["rel_path"].replace("\\", "/")
            if file_rel == candidate_suffix or file_rel.endswith("/" + candidate_suffix):
                return item["rel_path"]

        return ""

    def build(self) -> nx.DiGraph:
        """
        Resolves imports for all files and adds directed edges to the graph.
        """
        resolvers = {
            "python": self.resolve_python_import,
            "javascript": self.resolve_js_ts_import,
            "typescript": self.resolve_js_ts_import,
            "java": self.resolve_java_import,
        }

        for item in self.inventory:
            rel_path = item["rel_path"]
            language = item["language"]
            resolver = resolvers.get(language)
            if resolver is None:
                continue

            imports = self.parsed_data.get(rel_path, {}).get("imports", [])

            for imp in imports:
                resolved_path = resolver(rel_path, imp)

                if resolved_path and resolved_path != rel_path:
                    self.graph.add_edge(rel_path, resolved_path)

        return self.graph

    def to_mermaid(self) -> str:
        """
        Generates a Mermaid.js flowchart representation of the module dependency graph.
        """
        lines = ["graph TD"]
        
        # Define nodes with labels to display clean relative paths
        node_ids = {}
        for idx, node in enumerate(self.graph.nodes):
            node_id = f"node_{idx}"
            node_ids[node] = node_id
            lines.append(f'    {node_id}["{node}"]')
            
        # Define connections
        for u, v in self.graph.edges:
            lines.append(f"    {node_ids[u]} --> {node_ids[v]}")
            
        return "\n".join(lines)
