"""
Retrieval mechanism for LoCoBench.

This module provides retrieval-augmented generation (RAG) capabilities
for hard and expert difficulty scenarios. The retriever ranks full project
files by semantic similarity to the task prompt and returns only the top
percentage of files (default: 5%) to keep the final prompt compact.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not available. Retrieval will fall back to keyword-based method."
    )

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from sentence_transformers import SentenceTransformer as _SentenceTransformerType


DEFAULT_CODE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".kt",
    ".kts",
    ".md",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".sql",
    ".swift",
    ".ts",
    ".tsx",
    ".xml",
    ".yaml",
    ".yml",
}

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "dist",
    "build",
}

MAX_FILE_SIZE_BYTES = 800_000  # ~800 KB safeguard against huge binaries

MODEL_CACHE: Dict[str, "SentenceTransformer"] = {}


def split_code(code: str, chunk_size: int = 512) -> List[str]:
    """Split code into chunks for keyword-based fallback retrieval."""
    if not code:
        return []

    lines = code.splitlines()
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1  # include newline
        if current_size + line_size > chunk_size and current_chunk:
            chunks.append("\\n".join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size

    if current_chunk:
        chunks.append("\\n".join(current_chunk))

    return chunks


def _split_file_into_chunks(
    file_path: str,
    file_content: str,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Split a file into overlapping chunks with metadata.
    
    Args:
        file_path: Path to the file
        file_content: Full content of the file
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of chunk dictionaries with path, content, start_pos, end_pos, chunk_index
    """
    if not file_content:
        return []
    
    chunks: List[Dict[str, Any]] = []
    content_length = len(file_content)
    
    # For small files, return as single chunk
    if content_length <= chunk_size:
        return [{
            "path": file_path,
            "content": file_content,
            "start_pos": 0,
            "end_pos": content_length,
            "chunk_index": 0,
            "length": content_length,
        }]
    
    # Split into overlapping chunks
    start = 0
    chunk_index = 0
    
    while start < content_length:
        end = min(start + chunk_size, content_length)
        chunk_content = file_content[start:end]
        
        chunks.append({
            "path": file_path,
            "content": chunk_content,
            "start_pos": start,
            "end_pos": end,
            "chunk_index": chunk_index,
            "length": len(chunk_content),
        })
        
        chunk_index += 1
        # Move start position with overlap
        start = end - overlap if end < content_length else end
    
    return chunks


def _rank_chunks_with_embeddings(
    model: "SentenceTransformer",
    task_prompt: str,
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank code chunks using cosine similarity in embedding space."""
    if not chunks:
        return []
    
    texts = [task_prompt] + [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    
    query_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]
    
    similarities = np.dot(chunk_embeddings, query_embedding)
    
    for idx, chunk in enumerate(chunks):
        chunk["similarity"] = float(similarities[idx])
    
    return sorted(chunks, key=lambda c: c.get("similarity", 0.0), reverse=True)


def _normalize_relative_path(raw_path: str) -> str:
    """Normalise file paths to POSIX-style relative strings."""
    path = raw_path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.lstrip("/")


def _load_embedding_model(
    local_model_path: Optional[str],
    model_name: str,
) -> Optional["SentenceTransformer"]:
    """Load a SentenceTransformer model from cache, local path or HuggingFace hub."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    if local_model_path:
        cache_key = f"path::{Path(local_model_path).expanduser().resolve()}"
    else:
        cache_key = f"name::{model_name}"

    cached = MODEL_CACHE.get(cache_key)
    if cached:
        return cached

    try:
        if local_model_path:
            resolved = Path(local_model_path).expanduser().resolve()
            if resolved.exists():
                logger.info("Loading retrieval model from local path: %s", resolved)
                MODEL_CACHE[cache_key] = SentenceTransformer(str(resolved))
                return MODEL_CACHE[cache_key]
            logger.warning("Local retrieval model path does not exist: %s", resolved)

        logger.info("Loading retrieval model by name: %s", model_name)
        MODEL_CACHE[cache_key] = SentenceTransformer(model_name)
        return MODEL_CACHE[cache_key]
    except Exception as exc:  # pragma: no cover - dependent on external model availability
        logger.error("Failed to load retrieval model (%s): %s", cache_key, exc, exc_info=True)
        return None


def _is_supported_code_file(path: Path) -> bool:
    return path.suffix.lower() in DEFAULT_CODE_EXTENSIONS


def _collect_project_code_files(project_dir: Path) -> List[Dict[str, Any]]:
    """Traverse project directory and collect candidate code files."""
    if not project_dir or not project_dir.exists():
        return []

    candidates: List[Dict[str, Any]] = []
    base_dir = project_dir.resolve()

    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES and not d.startswith(".")]

        for filename in files:
            if filename.startswith("."):
                continue

            file_path = Path(root) / filename
            if not _is_supported_code_file(file_path):
                continue

            try:
                size = file_path.stat().st_size
            except OSError:
                continue

            if size == 0 or size > MAX_FILE_SIZE_BYTES:
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            candidates.append(
                {
                    "path": file_path.relative_to(base_dir).as_posix(),
                    "content": content,
                    "length": len(content),
                    "size_bytes": size,
                }
            )

    return candidates


def _prepare_candidate_files(
    context_files: Optional[Dict[str, str]],
    project_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    """Combine project files and scenario-provided context into a unified candidate set."""
    combined: Dict[str, Dict[str, Any]] = {}

    if project_dir and project_dir.exists():
        for file_info in _collect_project_code_files(project_dir):
            combined[file_info["path"]] = file_info

    if context_files:
        for raw_path, content in context_files.items():
            if not isinstance(content, str):
                continue

            normalized_path = _normalize_relative_path(str(raw_path))
            combined[normalized_path] = {
                "path": normalized_path,
                "content": content,
                "length": len(content),
                "size_bytes": len(content.encode("utf-8", errors="ignore")),
            }

    return list(combined.values())


def _rank_files_with_embeddings(
    model: "SentenceTransformer",
    task_prompt: str,
    candidate_files: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank candidate files using cosine similarity in embedding space."""
    if not candidate_files:
        return []

    texts = [task_prompt] + [file_info["content"] for file_info in candidate_files]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    query_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]

    similarities = np.dot(doc_embeddings, query_embedding)

    for idx, file_info in enumerate(candidate_files):
        file_info["similarity"] = float(similarities[idx])

    return sorted(candidate_files, key=lambda info: info.get("similarity", 0.0), reverse=True)


def _apply_length_budget(
    ranked_files: List[Dict[str, Any]],
    max_context_tokens: Optional[int],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Trim ranked files to satisfy the maximum context length (interpreted in characters).
    """
    if not ranked_files:
        return [], 0

    if not max_context_tokens or max_context_tokens <= 0:
        total_length = sum(max(info["length"], 0) for info in ranked_files)
        return ranked_files, total_length

    selected: List[Dict[str, Any]] = []
    running_total = 0

    for file_info in ranked_files:
        file_length = max(file_info["length"], 0)
        if not selected:
            selected.append(file_info)
            running_total += file_length
            continue

        if running_total + file_length <= max_context_tokens:
            selected.append(file_info)
            running_total += file_length
        else:
            logger.debug(
                "Skipping %s due to context cap (%d + %d > %d)",
                file_info["path"],
                running_total,
                file_length,
                max_context_tokens,
            )

    if not selected and ranked_files:
        top_file = ranked_files[0]
        return [top_file], max(top_file["length"], 0)

    return selected, running_total


def _format_retrieved_context(selected_files: List[Dict[str, Any]]) -> str:
    """Format selected files into a human-readable context block."""
    if not selected_files:
        return ""

    parts: List[str] = []
    for file_info in selected_files:
        similarity = file_info.get("similarity")
        similarity_part = (
            f"similarity: {similarity:.3f}, chars: {file_info['length']}"
            if similarity is not None
            else f"chars: {file_info['length']}"
        )
        header = f"### {file_info['path']} ({similarity_part})"
        parts.append(f"{header}\n```\n{file_info['content']}\n```")

    return "\n\n".join(parts)


def retrieve_relevant_embedding(
    context_files: Optional[Dict[str, str]],
    task_prompt: str,
    *,
    top_percent: float = 0.05,
    model_name: str = "all-MiniLM-L6-v2",
    project_dir: Optional[Path] = None,
    max_context_tokens: Optional[int] = None,
    local_model_path: Optional[str] = None,
    top_k: Optional[int] = None,
    smart_chunking: bool = True,
    chunks_per_file: int = 5,
    chunk_size: int = 2000,
) -> str:
    """
    Retrieve the most relevant project files using embeddings and return them as context.
    
    Args:
        smart_chunking: If True, split files into chunks and select most relevant chunks
        chunks_per_file: Maximum number of chunks to select per file (when smart_chunking=True)
        chunk_size: Size of each chunk in characters (when smart_chunking=True)
    """
    start_time = time.perf_counter()

    candidates = _prepare_candidate_files(context_files, project_dir)
    if not candidates:
        logger.warning("Retrieval: no candidate files found (project_dir=%s)", project_dir)
        return ""

    original_length_total = sum(max(info["length"], 0) for info in candidates)

    model = _load_embedding_model(local_model_path, model_name)
    if not model:
        logger.error("Retrieval: embedding model unavailable; aborting embedding-based retrieval.")
        return ""

    if top_percent is None or top_percent <= 0:
        selected_count = max(1, top_k or 1)
    else:
        selected_count = max(1, math.ceil(len(candidates) * top_percent))
    selected_count = min(selected_count, len(candidates))
    
    # Ensure we get at least top_k files if specified and it's larger than top_percent selection
    if top_k and top_k > selected_count:
        selected_count = min(top_k, len(candidates))
    
    logger.debug(
        "Retrieval: selecting %d files from %d candidates (top_percent=%.2f, top_k=%s, smart_chunking=%s)",
        selected_count,
        len(candidates),
        top_percent,
        top_k,
        smart_chunking,
    )

    ranked_files = _rank_files_with_embeddings(model, task_prompt, candidates)
    selected_files = ranked_files[:selected_count]

    if smart_chunking:
        # Split files into chunks and rank chunks by relevance
        all_chunks: List[Dict[str, Any]] = []
        
        for file_info in selected_files:
            file_chunks = _split_file_into_chunks(
                file_info["path"],
                file_info["content"],
                chunk_size=chunk_size,
                overlap=min(200, chunk_size // 10),  # 10% overlap or 200 chars, whichever is smaller
            )
            all_chunks.extend(file_chunks)
        
        logger.debug(
            "Retrieval: split %d files into %d chunks (avg %.1f chunks/file)",
            len(selected_files),
            len(all_chunks),
            len(all_chunks) / len(selected_files) if selected_files else 0,
        )
        
        # Rank all chunks by relevance
        ranked_chunks = _rank_chunks_with_embeddings(model, task_prompt, all_chunks)
        
        # Group chunks by file and select top chunks per file
        chunks_by_file: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in ranked_chunks:
            file_path = chunk["path"]
            if file_path not in chunks_by_file:
                chunks_by_file[file_path] = []
            chunks_by_file[file_path].append(chunk)
        
        # Select top chunks per file
        selected_chunks: List[Dict[str, Any]] = []
        for file_path, file_chunks in chunks_by_file.items():
            # Take top chunks_per_file chunks from each file
            top_chunks = sorted(file_chunks, key=lambda c: c.get("similarity", 0.0), reverse=True)[:chunks_per_file]
            selected_chunks.extend(top_chunks)
        
        # Re-rank selected chunks globally
        selected_chunks = sorted(selected_chunks, key=lambda c: c.get("similarity", 0.0), reverse=True)
        
        # Apply length budget to chunks
        trimmed_chunks, selected_length_total = _apply_length_budget(selected_chunks, max_context_tokens)
        
        # Group trimmed chunks back by file for formatting
        final_chunks_by_file: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in trimmed_chunks:
            file_path = chunk["path"]
            if file_path not in final_chunks_by_file:
                final_chunks_by_file[file_path] = []
            final_chunks_by_file[file_path].append(chunk)
        
        # Format output with chunk information
        parts: List[str] = []
        for file_path in sorted(final_chunks_by_file.keys()):
            file_chunks = final_chunks_by_file[file_path]
            # Sort chunks by position in file
            file_chunks.sort(key=lambda c: c.get("chunk_index", 0))
            
            # Calculate average similarity for this file
            avg_similarity = sum(c.get("similarity", 0.0) for c in file_chunks) / len(file_chunks)
            total_chars = sum(c.get("length", 0) for c in file_chunks)
            
            # Determine comment style based on file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext in {'.py', '.sh', '.rb', '.pl'}:
                comment_prefix = '#'
            elif file_ext in {'.java', '.js', '.ts', '.cpp', '.c', '.h', '.hpp', '.cs', '.go', '.rs', '.swift', '.kt'}:
                comment_prefix = '//'
            elif file_ext in {'.sql'}:
                comment_prefix = '--'
            else:
                comment_prefix = '//'  # Default to // style
            
            # Build content from chunks (with markers)
            chunk_contents = []
            for i, chunk in enumerate(file_chunks):
                chunk_idx = chunk.get("chunk_index", 0)
                start_pos = chunk.get("start_pos", 0)
                end_pos = chunk.get("end_pos", 0)
                similarity = chunk.get("similarity", 0.0)
                
                # Add chunk marker as comment
                chunk_marker = f"{comment_prefix} [Chunk {chunk_idx}: chars {start_pos}-{end_pos}, similarity: {similarity:.3f}]\n"
                chunk_contents.append(chunk_marker + chunk["content"])
                
                # Add separator between chunks (except for the last one)
                if i < len(file_chunks) - 1:
                    chunk_contents.append(f"\n{comment_prefix} ... [continues] ...\n")
            
            header = f"### {file_path} (avg similarity: {avg_similarity:.3f}, {len(file_chunks)} chunks, {total_chars} chars)"
            parts.append(f"{header}\n```\n{''.join(chunk_contents)}\n```")
        
        result = "\n\n".join(parts)
        
        duration = time.perf_counter() - start_time
        reduction_pct = (
            100.0 * (1.0 - (selected_length_total / original_length_total))
            if original_length_total
            else 0.0
        )
        
        logger.info(
            "Retrieval summary (smart chunking): project_dir=%s | files=%d | chunks=%d/%d | chars %d -> %d (Δ %.1f%%) | time %.2fs | max_context=%s",
            project_dir,
            len(final_chunks_by_file),
            len(trimmed_chunks),
            len(all_chunks),
            original_length_total,
            selected_length_total,
            reduction_pct,
            duration,
            max_context_tokens if max_context_tokens else "unlimited",
        )
        
        return result
    else:
        # Original behavior: use full files
        trimmed_files, selected_length_total = _apply_length_budget(selected_files, max_context_tokens)

        duration = time.perf_counter() - start_time
        reduction_pct = (
            100.0 * (1.0 - (selected_length_total / original_length_total))
            if original_length_total
            else 0.0
        )

        logger.info(
            "Retrieval summary: project_dir=%s | candidates=%d | selected=%d | chars %d -> %d (Δ %.1f%%) | time %.2fs | max_context=%s",
            project_dir,
            len(candidates),
            len(trimmed_files),
            original_length_total,
            selected_length_total,
            reduction_pct,
            duration,
            max_context_tokens if max_context_tokens else "unlimited",
        )
        logger.debug(
            "Selected files: %s",
            [
                f"{info['path']} (sim={info.get('similarity', 0.0):.3f}, chars={info['length']})"
                for info in trimmed_files
            ],
        )

        return _format_retrieved_context(trimmed_files)


def retrieve_relevant_keyword(
    context_files: Dict[str, str],
    task_prompt: str,
    top_k: int = 5,
    *,
    top_percent: Optional[float] = None,
    chunk_size: int = 512,
) -> str:
    """Retrieve relevant code fragments using keyword matching (fallback strategy)."""
    if not context_files:
        logger.warning("No context files provided for keyword retrieval.")
        return ""

    if top_percent is not None:
        estimated_top_k = max(1, math.ceil(len(context_files) * top_percent))
        top_k = max(top_k, estimated_top_k)

    import re  # Local import to avoid cost when embeddings are used

    words = re.findall(r"\b[a-zA-Z]{4,}\b", task_prompt.lower())
    stop_words = {
        "that",
        "this",
        "with",
        "from",
        "file",
        "code",
        "function",
        "class",
        "method",
        "should",
        "must",
        "need",
        "implement",
    }
    keywords = [word for word in words if word not in stop_words][:10]

    if not keywords:
        logger.warning("Keyword retrieval: no informative keywords extracted, returning first files.")
        retrieved_parts: List[str] = []
        for file_path, code_content in list(context_files.items())[:top_k]:
            chunks = split_code(code_content, chunk_size=chunk_size)
            if chunks:
                retrieved_parts.append(f"From {file_path}:\n{chunks[0]}")
        return "\n\n".join(retrieved_parts)

    chunk_scores: List[int] = []
    chunk_info: List[Tuple[str, int, str]] = []

    for file_path, code_content in context_files.items():
        for idx, chunk in enumerate(split_code(code_content, chunk_size=chunk_size)):
            chunk_lower = chunk.lower()
            score = sum(1 for keyword in keywords if keyword in chunk_lower)
            chunk_scores.append(score)
            chunk_info.append((file_path, idx, chunk))

    if not chunk_scores:
        return ""

    top_indices = np.argsort(chunk_scores)[-top_k:][::-1]

    retrieved_parts: List[str] = []
    for idx in top_indices:
        file_path, chunk_idx, chunk_content = chunk_info[idx]
        score = chunk_scores[idx]
        retrieved_parts.append(
            f"From {file_path} (chunk {chunk_idx + 1}, keyword matches: {score}):\n{chunk_content}"
        )

    logger.info("Retrieved %d fragments using keyword fallback.", len(top_indices))
    return "\n\n".join(retrieved_parts)


def retrieve_relevant(
    context_files: Optional[Dict[str, str]],
    task_prompt: str,
    top_k: int = 5,
    method: str = "embedding",
    model_name: str = "all-MiniLM-L6-v2",
    *,
    project_dir: Optional[Path] = None,
    top_percent: float = 0.05,
    max_context_tokens: Optional[int] = None,
    local_model_path: Optional[str] = None,
    chunk_size: int = 512,
    smart_chunking: bool = True,
    chunks_per_file: int = 5,
    retrieval_chunk_size: int = 2000,
) -> str:
    """Dispatch to the configured retrieval method."""
    if method == "embedding":
        result = retrieve_relevant_embedding(
            context_files or {},
            task_prompt,
            top_percent=top_percent,
            model_name=model_name,
            project_dir=project_dir,
            max_context_tokens=max_context_tokens,
            local_model_path=local_model_path,
            top_k=top_k,
            smart_chunking=smart_chunking,
            chunks_per_file=chunks_per_file,
            chunk_size=retrieval_chunk_size,
        )
        if not result and context_files:
            logger.warning("Embedding retrieval failed; falling back to keyword method.")
            return retrieve_relevant_keyword(
                context_files,
                task_prompt,
                top_k=top_k,
                top_percent=top_percent,
                chunk_size=chunk_size,
            )
        return result

    if method == "keyword":
        return retrieve_relevant_keyword(
            context_files or {},
            task_prompt,
            top_k=top_k,
            top_percent=top_percent,
            chunk_size=chunk_size,
        )

    logger.warning("Unknown retrieval method '%s'; defaulting to keyword fallback.", method)
    return retrieve_relevant_keyword(
        context_files or {},
        task_prompt,
        top_k=top_k,
        top_percent=top_percent,
        chunk_size=chunk_size,
    )


def load_context_files_from_scenario(
    scenario: Dict[str, Any],
    project_dir: Optional[Path] = None,
    include_all_project_files: bool = False,
) -> Dict[str, str]:
    """Load context file contents from a scenario definition."""
    context_obj = scenario.get("context_files")

    if isinstance(context_obj, dict):
        return {
            _normalize_relative_path(path): content
            for path, content in context_obj.items()
            if isinstance(content, str)
        }

    if include_all_project_files and project_dir and project_dir.exists():
        logger.info(
            "Loading full project files for scenario %s",
            scenario.get("id", "unknown"),
        )
        return {
            file_info["path"]: file_info["content"]
            for file_info in _collect_project_code_files(project_dir)
        }

    if not context_obj:
        logger.warning("Scenario %s does not provide context files.", scenario.get("id", "unknown"))
        return {}

    if not project_dir or not project_dir.exists():
        logger.warning(
            "Cannot load context files for scenario %s: project directory not provided or missing.",
            scenario.get("id", "unknown"),
        )
        return {}

    loaded_files: Dict[str, str] = {}
    for raw_path in context_obj:
        normalized_path = _normalize_relative_path(str(raw_path))
        candidate = project_dir / normalized_path

        if candidate.exists():
            try:
                loaded_files[normalized_path] = candidate.read_text(encoding="utf-8", errors="ignore")
                continue
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to read %s: %s", candidate, exc)

        logger.debug("Context file %s not found relative to %s", raw_path, project_dir)

    if not loaded_files:
        logger.warning(
            "Unable to resolve any context files for scenario %s within %s.",
            scenario.get("id", "unknown"),
            project_dir,
        )

    return loaded_files


__all__ = [
    "retrieve_relevant",
    "retrieve_relevant_embedding",
    "retrieve_relevant_keyword",
    "load_context_files_from_scenario",
    "split_code",
]
