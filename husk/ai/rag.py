import os
import json
import math
import hashlib
import re
import urllib.request
import urllib.error
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple, Set
from husk.ai.adapters import get_ssl_context

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Filtered out of the *query* only (not chunk text) so natural-language noise words in a
# question ("what does the ... do") don't dilute the overlap score against every chunk that
# happens to contain "the" or "do" — which, in code, is nearly all of them.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "what", "which", "who", "whom", "this", "that",
    "these", "those", "of", "in", "on", "at", "to", "for", "with", "by", "as",
    "it", "its", "how", "why", "when", "where", "and", "or", "but", "not", "if",
    "then", "so", "can", "could", "would", "should", "will", "shall", "i", "you",
    "he", "she", "we", "they", "them", "his", "her", "their", "our", "your",
}


def _tokenize(text: str, filter_stopwords: bool = False) -> List[str]:
    """
    Splits text/code into lowercase identifier-like tokens for keyword scoring.
    Deliberately simple (no stemming) since code identifiers are the signal that matters
    here (function/class/variable names), not natural-language grammar.
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if filter_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens

# Chunks larger than this many lines are split into overlapping sub-chunks so a single
# oversized class/function can't blow past embedding/LLM context limits.
MAX_CHUNK_LINES = 120
CHUNK_OVERLAP_LINES = 15

class SyntaxAwareChunker:
    """
    Splits codebase source files into logical chunks based on AST class and function ranges.
    """
    @staticmethod
    def _emit(
        chunks: List[Dict[str, Any]],
        rel_path: str,
        chunk_type: str,
        name: str,
        start: int,
        end: int,
        get_lines,
        label: str,
    ):
        """
        Appends one chunk for [start, end], transparently splitting it into overlapping
        sub-chunks if it exceeds MAX_CHUNK_LINES.
        """
        total_lines = end - start + 1
        if total_lines <= MAX_CHUNK_LINES:
            text = get_lines(start, end)
            if text.strip():
                chunks.append({
                    "text": f"// {label} in {rel_path}\n{text}",
                    "metadata": {
                        "rel_path": rel_path,
                        "type": chunk_type,
                        "name": name,
                        "start_line": start,
                        "end_line": end,
                    }
                })
            return

        part = 1
        cursor = start
        step = MAX_CHUNK_LINES - CHUNK_OVERLAP_LINES
        while cursor <= end:
            sub_end = min(cursor + MAX_CHUNK_LINES - 1, end)
            text = get_lines(cursor, sub_end)
            if text.strip():
                chunks.append({
                    "text": f"// {label} in {rel_path} (part {part})\n{text}",
                    "metadata": {
                        "rel_path": rel_path,
                        "type": chunk_type,
                        "name": f"{name} (part {part})",
                        "start_line": cursor,
                        "end_line": sub_end,
                    }
                })
            if sub_end >= end:
                break
            cursor += step
            part += 1

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
            if get_lines(start, end).strip():
                SyntaxAwareChunker._emit(chunks, rel_path, "class", cls["name"], start, end, get_lines, "Class")
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
                if get_lines(start, end).strip():
                    SyntaxAwareChunker._emit(chunks, rel_path, "function", func["name"], start, end, get_lines, "Function")
                    chunked_intervals.append((start, end))

        # 3. Chunk any remaining global code blocks (e.g. imports, setup, etc.)
        # Sort chunked intervals to find gaps
        sorted_intervals = sorted(chunked_intervals, key=lambda x: x[0])
        current_line = 1

        for start, end in sorted_intervals:
            if start > current_line:
                # Gap detected
                if get_lines(current_line, start - 1).strip():
                    SyntaxAwareChunker._emit(chunks, rel_path, "module_level", "global", current_line, start - 1, get_lines, "Module level code")
            current_line = max(current_line, end + 1)

        if current_line <= len(lines):
            if get_lines(current_line, len(lines)).strip():
                SyntaxAwareChunker._emit(chunks, rel_path, "module_level", "global", current_line, len(lines), get_lines, "Module level code")

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
        # Maps rel_path -> content SHA256, so re-indexing can skip unchanged files
        # instead of re-embedding the whole repo on every `husk ask` call.
        self.file_hashes: Dict[str, str] = {}
        self.load()

    def load(self):
        """
        Loads the index data from disk. Understands both the current
        {"chunks": [...], "file_hashes": {...}} format and the legacy flat-list format.
        """
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.chunks = data.get("chunks", [])
                    self.file_hashes = data.get("file_hashes", {})
                elif isinstance(data, list):
                    # Legacy format: a bare list of chunks with no per-file hash tracking.
                    self.chunks = data
                    self.file_hashes = {}
                else:
                    self.chunks, self.file_hashes = [], {}
            except Exception:
                self.chunks, self.file_hashes = [], {}
        else:
            self.chunks, self.file_hashes = [], {}

    def save(self):
        """
        Saves the index data to disk.
        """
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(self.index_path, "w") as f:
                json.dump({"chunks": self.chunks, "file_hashes": self.file_hashes}, f, indent=2)
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
        self.file_hashes = {}
        self.save()

    def get_file_hash(self, rel_path: str) -> Optional[str]:
        """
        Returns the content hash the index last saw for this file, or None if unseen.
        """
        return self.file_hashes.get(rel_path)

    def remove_file_chunks(self, rel_path: str):
        """
        Drops all chunks previously indexed for this file (used before re-indexing it,
        or when the file no longer exists in the repo).
        """
        self.chunks = [c for c in self.chunks if c.get("metadata", {}).get("rel_path") != rel_path]

    def update_file(self, rel_path: str, sha256: str, new_chunks: List[Dict[str, Any]]):
        """
        Replaces all chunks for a file with new_chunks (each a {"text", "metadata", "embedding"}
        dict) and records its content hash, so unchanged files can be skipped next time.
        """
        self.remove_file_chunks(rel_path)
        self.chunks.extend(new_chunks)
        self.file_hashes[rel_path] = sha256

    def remove_stale_files(self, known_rel_paths: Set[str]) -> List[str]:
        """
        Removes chunks/hashes for any indexed file that no longer exists in the repo.
        Returns the list of removed rel_paths.
        """
        stale = [p for p in self.file_hashes if p not in known_rel_paths]
        for p in stale:
            self.remove_file_chunks(p)
            del self.file_hashes[p]
        return stale

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

        This is a brute-force O(n) scan, which is fine up to a few thousand chunks.
        If this ever becomes a bottleneck on large repos, swap in an ANN index
        (e.g. hnswlib) behind this same method signature rather than reworking callers.
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

    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        vector_weight: float = 0.65,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Combines cosine-similarity vector search with an Okapi BM25 keyword score.

        Pure vector search can miss an exact identifier match (e.g. a specific class or
        function name in the query) when its embedding similarity doesn't happen to rank it
        in the top results. A naive "matches / chunk length" score doesn't reliably fix that:
        it over-rewards tiny chunks (a 2-line stub where one keyword hit is a huge fraction of
        its length beats a 500-token chunk with 8 hits of the same term) and under-weights
        common terms (on a codebase, words like "query" or "get" appear in a large fraction of
        chunks, so raw overlap barely differentiates on them). BM25 fixes both: term frequency
        saturates (a 9th mention of a word barely matters more than the 8th), and `bm25_b`
        controls length normalization against the *average* chunk length.

        `bm25_b` defaults to 0 (no length normalization) rather than the classic prose-tuned
        0.75: our chunks are already capped at MAX_CHUNK_LINES, so length differences here are
        a chunking artifact (a 120-line class chunk vs. a 2-line stub), not verbosity/padding —
        measured on real code, a nonzero `b` systematically pushed the most relevant (larger,
        information-dense) chunk out of the top results in favor of small stub classes that
        happened to contain a keyword once.

        Returns result dicts `{"chunk", "score", "vector_score", "keyword_score"}`, sorted
        by the blended `score` descending.
        """
        query_terms = set(_tokenize(query_text, filter_stopwords=True))

        if not self.chunks:
            return []

        # Tokenize every chunk once, and compute document frequency only for the terms
        # that actually appear in the query (cheap: a handful of terms, not the whole vocab).
        chunk_token_lists = [_tokenize(c.get("text", "")) for c in self.chunks]
        chunk_lengths = [len(tokens) for tokens in chunk_token_lists]
        avg_length = (sum(chunk_lengths) / len(chunk_lengths)) if chunk_lengths else 1.0

        doc_freq = {t: 0 for t in query_terms}
        for tokens in chunk_token_lists:
            present = set(tokens)
            for t in query_terms:
                if t in present:
                    doc_freq[t] += 1

        n_chunks = len(self.chunks)
        # Standard BM25 IDF (Robertson-Sparck Jones, +1 smoothed so it stays non-negative
        # even for terms present in most chunks).
        idf = {
            t: math.log((n_chunks - doc_freq[t] + 0.5) / (doc_freq[t] + 0.5) + 1.0)
            for t in query_terms
        }

        scored = []
        for chunk, chunk_tokens, length in zip(self.chunks, chunk_token_lists, chunk_lengths):
            emb = chunk.get("embedding")
            if not emb:
                continue
            vector_score = self._cosine_similarity(query_embedding, emb)

            keyword_score = 0.0
            if query_terms and chunk_tokens:
                chunk_counter = Counter(chunk_tokens)
                length_norm = 1 - bm25_b + bm25_b * (length / avg_length if avg_length else 1.0)
                for t in query_terms:
                    tf = chunk_counter.get(t, 0)
                    if tf == 0:
                        continue
                    keyword_score += idf[t] * (tf * (bm25_k1 + 1)) / (tf + bm25_k1 * length_norm)

            scored.append({"chunk": chunk, "vector_score": vector_score, "keyword_score": keyword_score})

        if not scored:
            return []

        # Min-max normalize keyword scores across this candidate set so they're on a
        # comparable scale to cosine similarity before blending.
        kw_values = [s["keyword_score"] for s in scored]
        kw_min, kw_max = min(kw_values), max(kw_values)
        kw_range = kw_max - kw_min

        for s in scored:
            norm_kw = (s["keyword_score"] - kw_min) / kw_range if kw_range > 0 else 0.0
            s["score"] = vector_weight * s["vector_score"] + (1 - vector_weight) * norm_kw

        scored.sort(key=lambda s: s["score"], reverse=True)
        return scored[:top_k]

class QueryEmbeddingCache:
    """
    Caches query embeddings on disk keyed by (model, query text) so repeated or iterative
    `husk ask` calls with the same question don't re-pay for the same embedding call.
    """
    def __init__(self, repo_path: str):
        self.cache_path = os.path.join(os.path.abspath(repo_path), ".husk", "query_cache.json")
        self.data: Dict[str, List[float]] = {}
        self.load()

    def load(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}
        else:
            self.data = {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def _key(model: Optional[str], text: str) -> str:
        hasher = hashlib.sha256()
        hasher.update((model or "").encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(text.encode("utf-8"))
        return hasher.hexdigest()

    def get(self, model: Optional[str], text: str) -> Optional[List[float]]:
        return self.data.get(self._key(model, text))

    def set(self, model: Optional[str], text: str, embedding: List[float]):
        self.data[self._key(model, text)] = embedding
        self.save()
