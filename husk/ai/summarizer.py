import os
import hashlib
from typing import List, Dict, Tuple, Optional, Callable, Any
from husk.ai.cache import FileCache
from husk.ai.adapters import BaseAdapter
from husk.ai.estimator import TokenEstimator

class CodeSummarizer:
    """
    Manages the Map-Reduce pipeline for codebase summarization:
    1. Map: Summarizes individual files (Leaf summaries).
    2. Reduce 1: Rollups file summaries into module summaries.
    3. Reduce 2: Synthesizes module summaries into system-level architecture documents.
    """
    def __init__(self, repo_path: str, adapter: BaseAdapter, cache: FileCache):
        self.repo_path = os.path.abspath(repo_path)
        self.adapter = adapter
        self.cache = cache
        self.estimator = TokenEstimator(adapter.model)

    def group_files_by_directory(self, inventory: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Groups source files by their parent directory path.
        """
        groups = {}
        for item in inventory:
            rel_path = item["rel_path"]
            dir_path = os.path.dirname(rel_path)
            # Standardize empty directory (root files) as "."
            if not dir_path:
                dir_path = "."
                
            if dir_path not in groups:
                groups[dir_path] = []
            groups[dir_path].append(rel_path)
        return groups

    def get_directory_combined_hash(self, rel_dir: str, child_files: List[str]) -> str:
        """
        Calculates a combined SHA256 hash of all child files in a directory.
        """
        hashes = []
        for file in sorted(child_files):
            full_path = os.path.join(self.repo_path, file)
            sha = self.cache.compute_sha256(full_path)
            if sha:
                hashes.append(sha)
                
        # Hash of concatenated child hashes plus directory path to prevent collisions
        hasher = hashlib.sha256()
        hasher.update(rel_dir.encode("utf-8"))
        hasher.update("".join(hashes).encode("utf-8"))
        return hasher.hexdigest()

    def summarize_file(self, rel_path: str, content: str, log_fn: Optional[Callable[[str], None]] = None) -> str:
        """
        Generates or retrieves a leaf-level summary of a file.
        """
        full_path = os.path.join(self.repo_path, rel_path)
        cached = self.cache.get_summary(rel_path, full_path)
        if cached:
            if log_fn:
                log_fn(f"[CACHE HIT] File: {rel_path}")
            return cached
            
        if log_fn:
            log_fn(f"[LLM CALL] File: {rel_path}")
            
        system_prompt = (
            "You are an expert software archaeologist. Summarize the following source code file in 2-3 concise sentences. "
            "Explain its main purpose, key responsibilities, and how it fits into the broader system."
        )
        user_prompt = f"File Path: {rel_path}\n\nCode Content:\n{content}"
        
        summary = self.adapter.generate(user_prompt, system_prompt).strip()
        self.cache.set_summary(rel_path, full_path, summary)
        return summary

    def summarize_directory(
        self, 
        rel_dir: str, 
        child_files: List[str], 
        file_summaries: Dict[str, str], 
        log_fn: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Rolls up individual file summaries into a module/directory summary.
        """
        combined_hash = self.get_directory_combined_hash(rel_dir, child_files)
        cached = self.cache.get_dir_summary(rel_dir, combined_hash)
        if cached:
            if log_fn:
                log_fn(f"[CACHE HIT] Module: {rel_dir}")
            return cached
            
        if log_fn:
            log_fn(f"[LLM CALL] Module: {rel_dir}")
            
        # Compile child summaries
        summaries_list = []
        for file in sorted(child_files):
            summary = file_summaries.get(file, "No summary available.")
            summaries_list.append(f"- {file}: {summary}")
            
        summaries_text = "\n".join(summaries_list)
        
        system_prompt = (
            "You are a principal software architect. Summarize the overall purpose, core features, "
            "and architectural responsibilities of the following codebase module (directory) in 2-3 concise sentences. "
            "Synthesize this module-level summary based solely on the individual file summaries provided."
        )
        user_prompt = f"Module Directory: {rel_dir}\n\nChild File Summaries:\n{summaries_text}"
        
        summary = self.adapter.generate(user_prompt, system_prompt).strip()
        self.cache.set_dir_summary(rel_dir, combined_hash, summary)
        return summary

    def generate_system_architecture(self, module_summaries: Dict[str, str]) -> str:
        """
        Compiles all module summaries into a system-level architecture document.
        """
        summaries_list = []
        for module, summary in sorted(module_summaries.items()):
            summaries_list.append(f"### Module: `{module}`\n{summary}\n")
            
        module_text = "\n".join(summaries_list)
        
        system_prompt = (
            "You are a principal software architect documenting a legacy codebase. "
            "Write a clean, professional, and comprehensive System Architecture Overview in Markdown format. "
            "Do NOT mention any missing API keys or run configs. Organize the output with headings, bullet points, "
            "and structured paragraphs. Include sections for: 1) System Overview, 2) High-Level Module Structure, "
            "and 3) Key Design Principles & Data Flows based on the module summaries."
        )
        user_prompt = f"Here are the module-level summaries for the codebase:\n\n{module_text}"
        
        return self.adapter.generate(user_prompt, system_prompt).strip()
