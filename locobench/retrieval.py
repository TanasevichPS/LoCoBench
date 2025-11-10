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
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, TYPE_CHECKING

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
                try:
                    # VESO models require trust_remote_code=True
                    MODEL_CACHE[cache_key] = SentenceTransformer(
                        str(resolved),
                        trust_remote_code=True
                    )
                    logger.info("Successfully loaded local retrieval model: %s", resolved)
                    return MODEL_CACHE[cache_key]
                except Exception as local_exc:
                    logger.error("Failed to load local model from %s: %s", resolved, local_exc, exc_info=True)
                    logger.warning("Falling back to model_name: %s", model_name)
            else:
                logger.warning("Local retrieval model path does not exist: %s (falling back to %s)", resolved, model_name)

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


def _extract_file_dependencies(file_path: str, file_content: str, project_dir: Optional[Path] = None) -> Set[str]:
    """
    Extract dependencies from a file (imports, includes, etc.).
    Returns set of internal file paths that this file depends on.
    Note: Returns potential dependency paths/names - actual resolution happens in _build_dependency_graph.
    """
    dependencies: Set[str] = set()
    
    if not file_content:
        return dependencies
    
    file_ext = Path(file_path).suffix.lower()
    base_dir = project_dir if project_dir else Path(file_path).parent
    
    # Java imports
    if file_ext == '.java':
        # Match: import com.example.ClassName; or import com.example.*;
        import_patterns = [
            r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s*;',
            r'import\s+static\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s*;',
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, file_content)
            for match in matches:
                # Skip java.lang and other standard library packages
                if match.startswith('java.') or match.startswith('javax.') or match.startswith('sun.'):
                    continue
                
                # Add both full package path and class name for flexible matching
                parts = match.split('.')
                if len(parts) > 0:
                    class_name = parts[-1]
                    # Add class name for simple matching
                    dependencies.add(class_name)
                    # Add package path variations
                    dependencies.add('/'.join(parts) + '.java')
                    dependencies.add('/'.join(parts[:-1]) + '/' + parts[-1] + '.java')
                    # Add common Java paths
                    dependencies.add('src/main/java/' + '/'.join(parts) + '.java')
                    dependencies.add('src/' + '/'.join(parts) + '.java')
    
    # Python imports
    elif file_ext == '.py':
        import_patterns = [
            r'^import\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            r'^from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
            r'^from\s+\.+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',  # Relative imports
        ]
        for pattern in import_patterns:
            matches = re.findall(pattern, file_content, re.MULTILINE)
            for match in matches:
                parts = match.split('.')
                if len(parts) > 0:
                    dependencies.add('/'.join(parts) + '.py')
                    dependencies.add('/'.join(parts[:-1]) + '/' + parts[-1] + '.py')
    
    # JavaScript/TypeScript imports
    elif file_ext in {'.js', '.ts', '.jsx', '.tsx'}:
        import_patterns = [
            r"import\s+.*\s+from\s+['\"]([^'\"]+)['\"]",
            r"import\s+['\"]([^'\"]+)['\"]",
            r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
        ]
        for pattern in import_patterns:
            matches = re.findall(pattern, file_content)
            for match in matches:
                # Skip node_modules and external packages
                if match.startswith('.') or '/' in match:
                    dependencies.add(match)
    
    # C/C++ includes
    elif file_ext in {'.c', '.cpp', '.h', '.hpp'}:
        include_patterns = [
            r'#include\s*["<]([^">]+)[">]',
        ]
        for pattern in include_patterns:
            matches = re.findall(pattern, file_content)
            for match in matches:
                dependencies.add(match)
    
    return dependencies


def _build_dependency_graph_fast(candidate_files: List[Dict[str, Any]], project_dir: Optional[Path] = None) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Fast version of dependency graph building - analyzes only first part of files for speed.
    """
    dependency_graph: Dict[str, Set[str]] = {}
    reverse_graph: Dict[str, Set[str]] = {}
    
    # Create path to file_info mapping (normalize paths)
    file_map: Dict[str, Dict[str, Any]] = {}
    for file_info in candidate_files:
        file_path = file_info["path"]
        normalized_path = _normalize_relative_path(file_path)
        file_map[normalized_path] = file_info
        dependency_graph[normalized_path] = set()
        reverse_graph[normalized_path] = set()
    
    # Also create mapping by filename (without extension) for better matching
    filename_map: Dict[str, List[str]] = {}
    for file_info in candidate_files:
        normalized_path = _normalize_relative_path(file_info["path"])
        filename = Path(normalized_path).stem
        if filename not in filename_map:
            filename_map[filename] = []
        filename_map[filename].append(normalized_path)
    
    # Extract dependencies for each file (use only first 2000 chars for speed)
    for file_info in candidate_files:
        file_path = file_info["path"]
        normalized_path = _normalize_relative_path(file_path)
        # Use only first 2000 chars for dependency extraction (much faster)
        content = file_info.get("content", "")[:2000]
        
        deps = _extract_file_dependencies(file_path, content, project_dir)
        
        # Resolve dependencies to internal files
        internal_deps: Set[str] = set()
        file_ext = Path(file_path).suffix.lower()
        
        for dep_path in deps:
            normalized_dep = _normalize_relative_path(dep_path)
            
            # Try exact match first
            if normalized_dep in file_map:
                internal_deps.add(normalized_dep)
            else:
                # For Java: try matching by class name (last part of package)
                if file_ext == '.java':
                    # Check if dep_path is just a class name (no path separators, no .java extension)
                    if '/' not in dep_path and not dep_path.endswith('.java'):
                        # This is a class name, try to find matching file
                        if dep_path in filename_map:
                            if len(filename_map[dep_path]) == 1:
                                internal_deps.add(filename_map[dep_path][0])
                            else:
                                # Find best match
                                current_dir = str(Path(normalized_path).parent)
                                best_match = filename_map[dep_path][0]
                                best_score = 0
                                for candidate in filename_map[dep_path]:
                                    candidate_dir = str(Path(candidate).parent)
                                    if candidate_dir == current_dir:
                                        best_match = candidate
                                        break
                                    elif current_dir in candidate_dir or candidate_dir in current_dir:
                                        score = len(set(current_dir.split('/')) & set(candidate_dir.split('/')))
                                        if score > best_score:
                                            best_score = score
                                            best_match = candidate
                                internal_deps.add(best_match)
                
                # Try matching by filename
                dep_filename = Path(normalized_dep).stem
                if dep_filename in filename_map:
                    if len(filename_map[dep_filename]) == 1:
                        internal_deps.add(filename_map[dep_filename][0])
                    else:
                        # Find best match based on directory similarity
                        current_dir = str(Path(normalized_path).parent)
                        best_match = filename_map[dep_filename][0]
                        best_score = 0
                        for candidate in filename_map[dep_filename]:
                            candidate_dir = str(Path(candidate).parent)
                            if candidate_dir == current_dir:
                                best_match = candidate
                                break
                            elif current_dir in candidate_dir or candidate_dir in current_dir:
                                score = len(set(current_dir.split('/')) & set(candidate_dir.split('/')))
                                if score > best_score:
                                    best_score = score
                                    best_match = candidate
                        internal_deps.add(best_match)
        
        dependency_graph[normalized_path] = internal_deps
        
        # Build reverse graph
        for dep in internal_deps:
            reverse_graph[dep].add(normalized_path)
    
    return dependency_graph, reverse_graph


def _build_dependency_graph(candidate_files: List[Dict[str, Any]], project_dir: Optional[Path] = None) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Full version of dependency graph building - analyzes full file content.
    Use _build_dependency_graph_fast for better performance.
    """
    # For now, use fast version - can be extended if needed
    return _build_dependency_graph_fast(candidate_files, project_dir)


def _find_dependent_files(
    selected_files: List[Dict[str, Any]],
    candidate_files: List[Dict[str, Any]],
    dependency_graph: Dict[str, Set[str]],
    reverse_graph: Dict[str, Set[str]],
    max_dependent_files: int = 10,
    is_architectural_task: bool = False,
) -> List[Dict[str, Any]]:
    """
    Find files that depend on selected files or are depended upon by selected files.
    Uses multiple strategies to find dependencies.
    Returns list of file_info dicts.
    """
    # Normalize selected file paths
    selected_paths = {_normalize_relative_path(file_info["path"]) for file_info in selected_files}
    dependent_paths: Set[str] = set()
    
    # Strategy 1: Find files that depend on selected files (reverse dependencies)
    for selected_path in selected_paths:
        dependents = reverse_graph.get(selected_path, set())
        dependent_paths.update(dependents)
    
    # Strategy 2: Find files that selected files depend on (forward dependencies)
    for selected_path in selected_paths:
        dependencies = dependency_graph.get(selected_path, set())
        dependent_paths.update(dependencies)
    
    # Strategy 3 & 4: Co-location and similar-name heuristics (only for non-architectural tasks)
    # Architectural tasks already have good dependency coverage, so skip heuristics to avoid noise
    if not is_architectural_task:
        # Create file_map for lookups
        file_map = {_normalize_relative_path(file_info["path"]): file_info for file_info in candidate_files}
        
        # Strategy 3: Co-location heuristic - files in same directory (up to 20% of max_dependent_files)
        selected_dirs: Set[str] = set()
        for file_info in selected_files:
            file_path = _normalize_relative_path(file_info["path"])
            file_dir = str(Path(file_path).parent)
            selected_dirs.add(file_dir)
        
        same_dir_files = []
        for file_path, file_info in file_map.items():
            if file_path not in selected_paths and file_path not in dependent_paths:
                file_dir = str(Path(file_path).parent)
                if file_dir in selected_dirs:
                    same_dir_files.append(file_info)
        
        # Add top same-dir files by similarity if available
        if same_dir_files:
            same_dir_files.sort(key=lambda f: f.get("similarity", 0.0), reverse=True)
            same_dir_limit = max(1, int(max_dependent_files * 0.20))  # 20% of max
            for file_info in same_dir_files[:same_dir_limit]:
                file_path = _normalize_relative_path(file_info["path"])
                if file_path not in selected_paths:
                    dependent_paths.add(file_path)
        
        # Strategy 4: Similar-name heuristic - files with similar names (up to 15% of max_dependent_files)
        selected_stems: Set[str] = set()
        for file_info in selected_files:
            file_path = _normalize_relative_path(file_info["path"])
            stem = Path(file_path).stem.lower()
            selected_stems.add(stem)
        
        # Look for files with similar stems (e.g., RoomStore and RoomStoreRepository)
        similar_name_files = []
        for file_path, file_info in file_map.items():
            if file_path not in selected_paths and file_path not in dependent_paths:
                stem = Path(file_path).stem.lower()
                # Check if stem contains or is contained in any selected stem
                for selected_stem in selected_stems:
                    if (selected_stem in stem or stem in selected_stem) and stem != selected_stem:
                        similar_name_files.append(file_info)
                        break
        
        # Add top similar-name files
        if similar_name_files:
            similar_name_files.sort(key=lambda f: f.get("similarity", 0.0), reverse=True)
            similar_limit = max(1, int(max_dependent_files * 0.15))  # 15% of max
            for file_info in similar_name_files[:similar_limit]:
                file_path = _normalize_relative_path(file_info["path"])
                if file_path not in selected_paths:
                    dependent_paths.add(file_path)
    
    # Remove files that are already selected
    dependent_paths -= selected_paths
    
    # Create file_info dicts for dependent files from candidates
    file_map = {_normalize_relative_path(file_info["path"]): file_info for file_info in candidate_files}
    dependent_files = [file_map[path] for path in dependent_paths if path in file_map]
    
    # Sort by similarity if available, otherwise keep order
    dependent_files.sort(key=lambda f: f.get("similarity", 0.0), reverse=True)
    
    return dependent_files[:max_dependent_files]


def _identify_important_files(
    candidate_files: List[Dict[str, Any]],
    selected_files: List[Dict[str, Any]],
    max_important_files: int = 5,
) -> List[Dict[str, Any]]:
    """
    Identify important files that haven't been selected yet.
    Important files are those with:
    - Large size (significant functionality)
    - Important names (main, config, core, etc.)
    - High complexity indicators
    """
    selected_paths = {_normalize_relative_path(file_info["path"]) for file_info in selected_files}
    remaining_files = [f for f in candidate_files if _normalize_relative_path(f["path"]) not in selected_paths]
    
    if not remaining_files:
        return []
    
    # Score files by importance
    important_keywords = [
        'main', 'core', 'config', 'util', 'common', 'base', 'service',
        'manager', 'controller', 'model', 'api', 'factory', 'builder'
    ]
    
    scored_files: List[Tuple[float, Dict[str, Any]]] = []
    
    for file_info in remaining_files:
        score = 0.0
        file_path_lower = file_info["path"].lower()
        file_name_lower = Path(file_info["path"]).name.lower()
        
        # Size score (normalized)
        max_size = max(f.get("size_bytes", 0) for f in remaining_files) if remaining_files else 1
        size_score = (file_info.get("size_bytes", 0) / max_size) * 0.3
        score += size_score
        
        # Name importance score
        for keyword in important_keywords:
            if keyword in file_path_lower or keyword in file_name_lower:
                score += 0.2
                break
        
        # Content indicators (check first 500 chars)
        content_preview = file_info.get("content", "")[:500].lower()
        if any(indicator in content_preview for indicator in ['class ', 'interface ', 'public class', 'abstract class']):
            score += 0.1
        
        scored_files.append((score, file_info))
    
    # Sort by score and return top files
    scored_files.sort(key=lambda x: x[0], reverse=True)
    return [file_info for _, file_info in scored_files[:max_important_files]]


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

    # MULTI-LEVEL RETRIEVAL STRATEGY (adaptive based on task type):
    # Detect task type and adjust strategy accordingly
    
    task_prompt_lower = task_prompt.lower()
    
    # Detect task type
    is_architectural_task = any(
        keyword in task_prompt_lower 
        for keyword in ['architect', 'architecture', 'structure', 'design', 'pattern', 'component', 'module', 'merge', 'refactor', 'critique', 'evaluate design']
    )
    
    is_code_comprehension_task = any(
        keyword in task_prompt_lower
        for keyword in ['trace', 'understand', 'comprehension', 'follow', 'track', 'flow', 'discrepancy', 'why', 'how does', 'explain']
    )
    
    is_security_task = any(
        keyword in task_prompt_lower
        for keyword in ['security', 'audit', 'vulnerability', 'secure', 'safe', 'protection', 'exploit', 'attack']
    )
    
    is_feature_implementation_task = any(
        keyword in task_prompt_lower
        for keyword in ['implement', 'add', 'create', 'build', 'develop', 'feature', 'functionality', 'etag', 'conditional']
    )
    
    # Apply multipliers based on task type (after detection)
    original_selected_count = selected_count
    if is_architectural_task:
        # For architectural tasks, increase file count moderately (but not too aggressive)
        # This helps capture more architectural context without overwhelming the prompt
        architectural_multiplier = 1.25  # 25% more files (conservative increase)
        selected_count = int(selected_count * architectural_multiplier)
        selected_count = min(selected_count, len(candidates))
        logger.debug("🏗️ Architectural task: increased file count from %d to %d (%.1fx)", 
                    original_selected_count, selected_count, architectural_multiplier)
    elif is_code_comprehension_task:
        # Moderate increase for code comprehension to capture more flow context
        selected_count = int(selected_count * 1.15)  # 15% more files
        selected_count = min(selected_count, len(candidates))
        logger.debug("🔍 Code comprehension: increased file count from %d to %d (1.15x)", 
                    original_selected_count, selected_count)
    
    # Optimized adaptive ratios based on task type
    if is_architectural_task:
        # For architectural tasks: more dependencies for structure understanding
        level1_ratio = 0.55  # More semantic for better quality selection
        level2_ratio = 0.35  # More dependencies (structure matters)
        level3_ratio = 0.10  # Important files
        logger.debug("🏗️ Architectural task detected: L1=55%, L2=35%, L3=10%")
    elif is_code_comprehension_task:
        # For code comprehension: more dependencies for tracing flow
        level1_ratio = 0.65  # Good semantic coverage
        level2_ratio = 0.30  # More dependencies for tracing (increased from 25%)
        level3_ratio = 0.05  # Less important files (reduced from 10%)
        logger.debug("🔍 Code comprehension task detected: L1=65%, L2=30%, L3=5%")
    elif is_security_task:
        # For security: more semantic (find security-related code)
        level1_ratio = 0.70
        level2_ratio = 0.20  # Some dependencies for context
        level3_ratio = 0.10
        logger.debug("🔒 Security task detected: L1=70%, L2=20%, L3=10%")
    elif is_feature_implementation_task:
        # For feature implementation: more semantic (find relevant code to modify)
        level1_ratio = 0.75  # More semantic to find relevant code
        level2_ratio = 0.15  # Some dependencies for context
        level3_ratio = 0.10
        logger.debug("⚙️ Feature implementation task detected: L1=75%, L2=15%, L3=10%")
    else:
        # Default: balanced approach
        level1_ratio = 0.70
        level2_ratio = 0.20
        level3_ratio = 0.10
        logger.debug("📝 Default task: L1=70%, L2=20%, L3=10%")
    
    ranked_files = _rank_files_with_embeddings(model, task_prompt, candidates)
    
    # Boost architectural files BEFORE selection for architectural tasks
    if is_architectural_task:
        architectural_keywords = [
            'interface', 'abstract', 'base', 'config', 'main', 'entry', 
            'factory', 'builder', 'strategy', 'adapter', 'service', 'manager',
            'controller', 'model', 'view', 'util', 'common', 'core', 'api',
            'store', 'repository', 'worker', 'handler', 'processor',
            'room', 'offline', 'sync', 'data', 'persistence', 'cache', 'dao', 'dto', 'entity'
        ]
        
        # Extract words from task prompt for matching
        task_words = set(task_prompt_lower.split())
        
        boosted_count = 0
        for file_info in ranked_files:
            file_path_lower = file_info["path"].lower()
            file_name_lower = Path(file_info["path"]).name.lower()
            original_sim = file_info.get("similarity", 0.0)
            
            # Boost similarity for architectural files
            boost = 0.0
            
            # 1. Boost for architectural keywords in path/name (further increased)
            keyword_matches = sum(1 for keyword in architectural_keywords if keyword in file_path_lower or keyword in file_name_lower)
            if keyword_matches > 0:
                boost += 0.18 + (keyword_matches * 0.02)  # Base 0.18 + 0.02 per additional keyword (max ~0.30)
            
            # 2. Boost for architectural patterns in content (increased and extended scan)
            content_preview = file_info.get("content", "")[:2000]  # Extended to 2000 chars
            content_lower = content_preview.lower()
            architectural_patterns = [
                'interface ', 'abstract class', 'implements', 'extends', 'public class',
                'public interface', '@service', '@component', '@repository', '@entity',
                'class.*extends', 'class.*implements'
            ]
            pattern_matches = sum(1 for pattern in architectural_patterns if pattern in content_lower)
            if pattern_matches > 0:
                boost += 0.25 + (pattern_matches * 0.05)  # Base 0.25 + 0.05 per pattern
            
            # 3. Boost for files with high similarity already (they're likely relevant)
            if original_sim > 0.20:  # Lowered threshold from 0.25 to 0.20
                boost += 0.12  # Increased from 0.10
            
            # 4. Boost for files mentioned in task prompt (by name)
            file_words = set(file_name_lower.split('_') + file_name_lower.split('-') + [file_name_lower])
            common_words = task_words.intersection(file_words)
            if len(common_words) > 0:
                boost += 0.15  # Increased from 0.10
            
            # 5. Additional boost for entry points and configuration files
            if any(indicator in file_name_lower for indicator in ['main', 'application', 'config', 'factory', 'builder']):
                boost += 0.12  # Additional boost for entry points
            
            if boost > 0:
                file_info["similarity"] = min(1.0, original_sim + boost)
                boosted_count += 1
        
        # Re-rank after boosting
        ranked_files = sorted(ranked_files, key=lambda info: info.get("similarity", 0.0), reverse=True)
        logger.debug("🏗️ Boosted %d architectural files before selection (max boost applied)", boosted_count)
    
    # Calculate how many files to select at each level
    # Level 1: Top semantically relevant files
    level1_count = max(1, int(selected_count * level1_ratio))
    
    # For architectural tasks, apply quality filter - only select files with good similarity
    if is_architectural_task:
        # Filter to files with similarity > 0.12 (after boost) to ensure quality
        # This is a soft filter - we still want to capture architectural files even if similarity is moderate
        quality_threshold = 0.12
        quality_files = [f for f in ranked_files if f.get("similarity", 0.0) > quality_threshold]
        if len(quality_files) >= level1_count:
            level1_files = quality_files[:level1_count]
            logger.debug(
                "🏗️ Architectural quality filter: selected %d files with similarity > %.2f",
                len(level1_files),
                quality_threshold
            )
        else:
            # If not enough quality files, use all available but log warning
            level1_files = ranked_files[:level1_count]
            logger.debug(
                "🏗️ Architectural: only %d files meet quality threshold, using top %d",
                len(quality_files),
                level1_count
            )
    else:
        level1_files = ranked_files[:level1_count]
    
    logger.debug(
        "📊 Multi-level retrieval: Level 1 (semantic) selected %d files",
        len(level1_files)
    )
    
    # Level 2: Files with dependencies (adaptive based on task type)
    level2_count = max(0, int(selected_count * level2_ratio))
    dependency_files: List[Dict[str, Any]] = []
    
    # Only analyze dependencies if we have project_dir and it's worth it
    if level2_count > 0 and project_dir and len(candidates) > 5:
        try:
            # Build dependency graph once for all candidates
            # Use lightweight analysis: limit file content analysis to first 2000 chars for speed
            dependency_graph, reverse_graph = _build_dependency_graph_fast(candidates, project_dir)
            
            # Find dependent files (allow up to level2_count * 2 to have options)
            dependent_files = _find_dependent_files(
                level1_files,
                candidates,
                dependency_graph,
                reverse_graph,
                max_dependent_files=min(level2_count * 2, selected_count - level1_count),
                is_architectural_task=is_architectural_task,
            )
            
            # Limit to level2_count
            dependent_files = dependent_files[:level2_count]
            
            logger.debug(
                "📊 Multi-level retrieval: Level 2 (dependencies) found %d files",
                len(dependent_files)
            )
        except Exception as e:
            logger.debug("Dependency analysis skipped or failed: %s", e)
            dependent_files = []
    
    # Level 3: Beginning of other important files (adaptive based on task type)
    remaining_budget = selected_count - len(level1_files) - len(dependent_files)
    level3_count = max(0, min(remaining_budget, int(selected_count * level3_ratio)))
    important_files: List[Dict[str, Any]] = []
    
    if level3_count > 0:
        # Combine already selected files
        already_selected = level1_files + dependent_files
        important_files = _identify_important_files(
            candidates,
            already_selected,
            max_important_files=level3_count
        )
        
        logger.debug(
            "📊 Multi-level retrieval: Level 3 (important) selected %d files",
            len(important_files)
        )
    
    # If we didn't fill the budget, add more Level 1 files (quality over quantity)
    if len(level1_files) + len(dependent_files) + len(important_files) < selected_count:
        remaining = selected_count - len(level1_files) - len(dependent_files) - len(important_files)
        additional_level1 = ranked_files[len(level1_files):len(level1_files) + remaining]
        level1_files.extend(additional_level1)
        logger.debug(
            "📊 Multi-level retrieval: Added %d more Level 1 files to fill budget",
            len(additional_level1)
        )
    
    # Mark files with their level for later processing
    for file_info in level1_files:
        file_info["retrieval_level"] = 1
    for file_info in dependent_files:
        file_info["retrieval_level"] = 2
    for file_info in important_files:
        file_info["retrieval_level"] = 3
    
    # Combine all selected files
    selected_files = level1_files + dependent_files + important_files
    
    # Remove duplicates (in case a file appears in multiple levels)
    # Keep the file from the highest priority level (lower level number = higher priority)
    seen_paths: Dict[str, Dict[str, Any]] = {}
    for file_info in selected_files:
        file_path = _normalize_relative_path(file_info["path"])
        if file_path not in seen_paths:
            seen_paths[file_path] = file_info
        else:
            # Keep file from lower level (higher priority)
            current_level = seen_paths[file_path].get("retrieval_level", 999)
            new_level = file_info.get("retrieval_level", 999)
            if new_level < current_level:
                seen_paths[file_path] = file_info
    
    selected_files = list(seen_paths.values())
    
    logger.info(
        "📊 Multi-level retrieval summary: Level1=%d, Level2=%d, Level3=%d, Total=%d",
        len(level1_files),
        len(dependent_files),
        len(important_files),
        len(selected_files)
    )

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
        
        # Select top chunks per file with smart strategy:
        # 1. Always include the first chunk (file header, imports, class definitions)
        # 2. Then select most relevant chunks with diversification (spread across file)
        # 3. For architectural tasks, prioritize beginning chunks (class/interface definitions)
        selected_chunks: List[Dict[str, Any]] = []
        # Use is_architectural_task from outer scope (already defined above)
        
        # Get file level information
        file_level_map: Dict[str, int] = {}
        for file_info in selected_files:
            normalized_path = _normalize_relative_path(file_info["path"])
            file_level_map[normalized_path] = file_info.get("retrieval_level", 1)
        
        for file_path, file_chunks in chunks_by_file.items():
            # Normalize file path for lookup
            normalized_file_path = _normalize_relative_path(file_path)
            
            # Sort chunks by position in file to find first chunk
            file_chunks_sorted_by_pos = sorted(file_chunks, key=lambda c: c.get("chunk_index", 0))
            first_chunk = file_chunks_sorted_by_pos[0] if file_chunks_sorted_by_pos else None
            
            # Sort by relevance
            file_chunks_sorted_by_relevance = sorted(file_chunks, key=lambda c: c.get("similarity", 0.0), reverse=True)
            
            # Get retrieval level for this file
            file_level = file_level_map.get(normalized_file_path, 1)
            
            # Select chunks based on retrieval level
            top_chunks = []
            
            if file_level == 3:
                # Level 3: Only take beginning chunks (first 1-2 chunks)
                max_chunks_for_level3 = min(2, chunks_per_file)
                for i in range(min(max_chunks_for_level3, len(file_chunks_sorted_by_pos))):
                    top_chunks.append(file_chunks_sorted_by_pos[i])
                logger.debug(
                    "File %s (Level 3): selected first %d chunks only",
                    file_path,
                    len(top_chunks)
                )
            else:
                # Level 1 and 2: Use full smart chunking strategy
                # Select chunks: always include first chunk, then diversify
                if first_chunk:
                    top_chunks.append(first_chunk)
                
                # For architectural tasks, also prioritize first few chunks (class definitions)
                if is_architectural_task and len(file_chunks_sorted_by_pos) > 1:
                    # Include first 3-4 chunks if they exist (usually contain class/interface definitions)
                    for i in range(1, min(4, len(file_chunks_sorted_by_pos))):
                        early_chunk = file_chunks_sorted_by_pos[i]
                        if early_chunk not in top_chunks and len(top_chunks) < chunks_per_file:
                            # Stronger boost for early chunks in architectural tasks
                            early_chunk["similarity"] = early_chunk.get("similarity", 0.0) + 0.10
                            top_chunks.append(early_chunk)
                
                # Diversification strategy: select chunks from different parts of the file
                # Divide file into regions and try to get at least one chunk from each region
                if len(file_chunks_sorted_by_pos) > 1:
                    num_regions = min(3, chunks_per_file - len(top_chunks))  # Adjust based on already selected chunks
                    if num_regions > 0:
                        region_size = len(file_chunks_sorted_by_pos) // num_regions
                        
                        # Try to get one chunk from each region
                        for region_idx in range(num_regions):
                            region_start = region_idx * region_size
                            region_end = (region_idx + 1) * region_size if region_idx < num_regions - 1 else len(file_chunks_sorted_by_pos)
                            region_chunks = file_chunks_sorted_by_pos[region_start:region_end]
                            
                            # Find most relevant chunk in this region
                            if region_chunks:
                                region_chunks_by_relevance = sorted(region_chunks, key=lambda c: c.get("similarity", 0.0), reverse=True)
                                best_in_region = region_chunks_by_relevance[0]
                                
                                if best_in_region not in top_chunks and len(top_chunks) < chunks_per_file:
                                    top_chunks.append(best_in_region)
                
                # Fill remaining slots with top relevant chunks
                for chunk in file_chunks_sorted_by_relevance:
                    if len(top_chunks) >= chunks_per_file:
                        break
                    if chunk not in top_chunks:
                        top_chunks.append(chunk)
            
            selected_chunks.extend(top_chunks)
            
            logger.debug(
                "File %s: selected %d chunks (first chunk: %s, early chunks: %d, diversification: %s, top similarity: %.3f)",
                file_path,
                len(top_chunks),
                "yes" if first_chunk in top_chunks else "no",
                sum(1 for c in top_chunks if c.get("chunk_index", 0) < 3),
                "yes" if len(set(c.get("chunk_index", 0) for c in top_chunks)) > 2 else "no",
                file_chunks_sorted_by_relevance[0].get("similarity", 0.0) if file_chunks_sorted_by_relevance else 0.0,
            )
        
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
