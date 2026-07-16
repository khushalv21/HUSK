import os
import json
import math
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple
from husk.ai.adapters import get_ssl_context

class SyntaxAwareChunker:
    """
    Splits codebase source files into logical chunks based on AST class and function ranges.
    """
    @staticmethod
    def chunk_file(rel_path: str, content: str, parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        lines = content.splitlines()
        
        # Get definitions
        classes = parsed_data.get("classes", [])
        functions = parsed_data.get("functions", [])
        
        # Keep track of line intervals that have been chunked
        chunked_intervals: List[Tuple[int, int]] = []
        
        # Helper to extract lines (1-indexed)
        def get_lines(start: int, end: int) -> str:
            # Clamp lines
            s = max(1, start) - 1
            e = min(len(lines), end)
            return "\n".join(lines[s:e])

        # 1. Chunk class declarations
        for cls in classes:
            start, end = cls["start_line"], cls["end_line"]
            chunk_text = get_lines(start, end)
            if chunk_text.strip():
                chunks.append({
                    "text": f"// Class in {rel_path}\n{chunk_text}",
                    "metadata": {
                        "rel_path": rel_path,
                        "type": "class",
                        "name": cls["name"],
                        "start_line": start,
                        "end_line": end
                    }
                })
                chunked_intervals.append((start, end))

        # 2. Chunk functions/methods that are NOT nested inside any already-chunked classes
        for func in functions:
            start, end = func["start_line"], func["end_line"]
            
            # Check if this function falls inside any class interval
            inside_class = False
            for c_start, c_end in chunked_intervals:
                if start >= c_start and end <= c_end:
                    inside_class = True
                    break
                    
            if not inside_class:
                chunk_text = get_lines(start, end)
                if chunk_text.strip():
                    chunks.append({
                        "text": f"// Function in {rel_path}\n{chunk_text}",
                        "metadata": {
                            "rel_path": rel_path,
                            "type": "function",
                            "name": func["name"],
                            "start_line": start,
                            "end_line": end
                        }
                    })
                    chunked_intervals.append((start, end))
                    
        # 3. Chunk any remaining global code blocks (e.g. imports, setup, etc.)
        # Sort chunked intervals to find gaps
        sorted_intervals = sorted(chunked_intervals, key=lambda x: x[0])
        current_line = 1
        
        for start, end in sorted_intervals:
            if start > current_line:
                # Gap detected
                gap_text = get_lines(current_line, start - 1)
                if gap_text.strip():
                    chunks.append({
                        "text": f"// Module level code in {rel_path}\n{gap_text}",
                        "metadata": {
                            "rel_path": rel_path,
                            "type": "module_level",
                            "name": "global",
                            "start_line": current_line,
                            "end_line": start - 1
                        }
                    })
            current_line = max(current_line, end + 1)
            
        if current_line <= len(lines):
            gap_text = get_lines(current_line, len(lines))
            if gap_text.strip():
                chunks.append({
                    "text": f"// Module level code in {rel_path}\n{gap_text}",
                    "metadata": {
                        "rel_path": rel_path,
                        "type": "module_level",
                        "name": "global",
                        "start_line": current_line,
                        "end_line": len(lines)
                    }
                })
                
        return chunks

class EmbeddingClient:
    """
    Fetches text embeddings from OpenAI or Ollama.
    """
    def __init__(self, provider: str, api_key: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key
        
        if self.provider == "openai":
            self.model = model or "text-embedding-3-small"
        elif self.provider in ("ollama", "local"):
            self.model = model or "nomic-embed-text"
        else:
            self.model = model

    def get_embedding(self, text: str) -> List[float]:
        """
        Retrieves the float embedding vector for a given text.
        """
        if self.provider == "openai":
            if not self.api_key:
                raise ValueError("OpenAI API Key is required for embeddings.")
                
            url = "https://api.openai.com/v1/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": self.model,
                "input": text
            }
            
            req = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode("utf-8"), 
                headers=headers,
                method="POST"
            )
            
            try:
                with urllib.request.urlopen(req, timeout=15, context=get_ssl_context()) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    return res["data"][0]["embedding"]
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"OpenAI Embedding HTTP Error {e.code}: {err_body}")
            except Exception as e:
                raise RuntimeError(f"Failed to fetch OpenAI embedding: {e}")
                
        elif self.provider in ("ollama", "local"):
            host = self.api_key or "http://localhost:11434" # host stored in api_key parameter
            url = f"{host}/api/embeddings"
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            req = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode("utf-8"), 
                headers=headers,
                method="POST"
            )
            
            try:
                with urllib.request.urlopen(req, timeout=15, context=get_ssl_context()) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    # Ollama response maps embedding to the key "embedding"
                    return res["embedding"]
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"Ollama Embedding HTTP Error {e.code}: {err_body}")
            except Exception as e:
                raise RuntimeError(f"Failed to fetch Ollama embedding: {e}. Is Ollama running?")
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

class VectorIndex:
    """
    A lightweight, pure-Python vector database that stores chunks and embeddings.
    Serializes to a local JSON file.
    """
    def __init__(self, index_path: str):
        self.index_path = os.path.abspath(index_path)
        self.chunks: List[Dict[str, Any]] = []
        self.load()

    def load(self):
        """
        Loads the index data from disk.
        """
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r") as f:
                    self.chunks = json.load(f)
            except Exception:
                self.chunks = []
        else:
            self.chunks = []

    def save(self):
        """
        Saves the index data to disk.
        """
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(self.index_path, "w") as f:
                json.dump(self.chunks, f, indent=2)
        except Exception:
            pass

    def add_chunk(self, text: str, metadata: Dict[str, Any], embedding: List[float]):
        """
        Adds a new chunk to the index.
        """
        self.chunks.append({
            "text": text,
            "metadata": metadata,
            "embedding": embedding
        })

    def clear(self):
        """
        Clears the index.
        """
        self.chunks = []
        self.save()

    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """
        Calculates cosine similarity between two vectors.
        """
        if len(v1) != len(v2):
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude_v1 = math.sqrt(sum(a * a for a in v1))
        magnitude_v2 = math.sqrt(sum(b * b for b in v2))
        
        if magnitude_v1 * magnitude_v2 == 0.0:
            return 0.0
            
        return dot_product / (magnitude_v1 * magnitude_v2)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs a cosine similarity search against stored embeddings.
        Returns a sorted list of (chunk, similarity) tuples.
        """
        results = []
        for chunk in self.chunks:
            emb = chunk.get("embedding")
            if emb:
                similarity = self._cosine_similarity(query_embedding, emb)
                results.append((chunk, similarity))
                
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
