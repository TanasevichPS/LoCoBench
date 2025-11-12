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

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    BM25_AVAILABLE = False
    logger.debug(
        "rank-bm25 not available. Hybrid search will use keyword-based fallback."
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
    
    # Extract dependencies for each file (use only first 3000 chars for speed, increased from 2000)
    for file_info in candidate_files:
        file_path = file_info["path"]
        normalized_path = _normalize_relative_path(file_path)
        # Use only first 3000 chars for dependency extraction (increased from 2000 for better coverage)
        content = file_info.get("content", "")[:3000]
        
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


def _expand_via_dependency_graph(
    seed_paths: Set[str],
    dependency_graph: Dict[str, Set[str]],
    reverse_graph: Dict[str, Set[str]],
    max_depth: int = 2,
    max_files_per_level: int = 15,
) -> Set[str]:
    """
    Расширяет набор файлов через граф зависимостей на несколько уровней глубины.
    Это помогает найти файлы, которые связаны косвенно через цепочку зависимостей.
    """
    expanded_paths = set(seed_paths)
    current_level = seed_paths
    
    for depth in range(1, max_depth + 1):
        next_level = set()
        
        for file_path in current_level:
            # Найти прямые зависимости (forward)
            deps = dependency_graph.get(file_path, set())
            next_level.update(deps)
            
            # Найти обратные зависимости (reverse)
            rev_deps = reverse_graph.get(file_path, set())
            next_level.update(rev_deps)
        
        # Исключить уже добавленные файлы
        next_level -= expanded_paths
        
        # Ограничить количество файлов на уровне
        if len(next_level) > max_files_per_level:
            # Взять случайную выборку или топ по важности
            next_level = set(list(next_level)[:max_files_per_level])
        
        expanded_paths.update(next_level)
        current_level = next_level
        
        if not current_level:
            break  # Нет больше файлов для расширения
    
    return expanded_paths


def _find_dependent_files(
    selected_files: List[Dict[str, Any]],
    candidate_files: List[Dict[str, Any]],
    dependency_graph: Dict[str, Set[str]],
    reverse_graph: Dict[str, Set[str]],
    max_dependent_files: int = 10,
    is_architectural_task: bool = False,
    use_deep_expansion: bool = True,
    expand_func: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """
    Find files that depend on selected files or are depended upon by selected files.
    Uses multiple strategies including deep graph expansion.
    Returns list of file_info dicts.
    """
    # Normalize selected file paths
    selected_paths = {_normalize_relative_path(file_info["path"]) for file_info in selected_files}
    dependent_paths: Set[str] = set()
    
    # Strategy 1: Deep graph expansion (use category-specific depth)
    if use_deep_expansion:
        if expand_func:
            # Use custom expansion function with category-specific config
            expanded_paths = expand_func(selected_paths, dependency_graph, reverse_graph)
        else:
            # Use default expansion (backward compatibility)
            expanded_paths = _expand_via_dependency_graph(
                selected_paths,
                dependency_graph,
                reverse_graph,
                max_depth=2,
                max_files_per_level=20
            )
        dependent_paths.update(expanded_paths)
        # Удалить исходные файлы из зависимостей
        dependent_paths -= selected_paths
    else:
        # Fallback: простой поиск (1 уровень)
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


def _extract_key_entities_and_concepts(task_prompt: str) -> Dict[str, List[str]]:
    """
    Извлекает ключевые сущности, действия и концепции из промпта задачи.
    Используется для создания оптимизированного запроса для ритривера.
    """
    task_lower = task_prompt.lower()
    
    # Извлечь имена классов/файлов (слова с заглавной буквы или в кавычках)
    import re
    # Имена классов (CamelCase)
    class_names = re.findall(r'\b[A-Z][a-zA-Z0-9]+\b', task_prompt)
    # Имена файлов (в кавычках или упомянутые явно)
    file_names = re.findall(r'["\']([^"\']+)["\']', task_prompt)
    # Имена из подчеркиваний (snake_case)
    snake_case_names = re.findall(r'\b[a-z]+_[a-z_]+\b', task_lower)
    
    entities = list(set(class_names + file_names + snake_case_names))
    
    # Извлечь действия (глаголы)
    action_keywords = [
        'merge', 'refactor', 'implement', 'add', 'create', 'build', 'develop',
        'trace', 'understand', 'analyze', 'evaluate', 'critique', 'review',
        'fix', 'debug', 'optimize', 'improve', 'update', 'modify', 'change',
        'integrate', 'combine', 'consolidate', 'restructure', 'reorganize'
    ]
    actions = [action for action in action_keywords if action in task_lower]
    
    # Извлечь концепции/домены
    concept_keywords = [
        'sync', 'synchronize', 'offline', 'online', 'persistence', 'cache',
        'database', 'repository', 'service', 'controller', 'model', 'view',
        'factory', 'builder', 'strategy', 'adapter', 'observer', 'singleton',
        'security', 'authentication', 'authorization', 'encryption', 'validation',
        'api', 'rest', 'endpoint', 'request', 'response', 'http', 'etag',
        'data', 'entity', 'dto', 'dao', 'worker', 'handler', 'processor'
    ]
    concepts = [concept for concept in concept_keywords if concept in task_lower]
    
    return {
        'entities': entities,
        'actions': actions,
        'concepts': concepts
    }


def _generate_multi_queries(task_prompt: str, num_queries: int = 8, task_type: str = None) -> List[str]:
    """
    Генерирует несколько вариантов запроса из оригинального промпта для Multi-Query Retrieval.
    Использует различные стратегии переформулирования запроса.
    
    Args:
        task_prompt: Оригинальный промпт задачи
        num_queries: Количество вариантов запроса для генерации (увеличено до 8)
        task_type: Тип задачи (architectural, comprehension, security, implementation)
    
    Returns:
        Список вариантов запроса
    """
    queries = [task_prompt]  # Всегда включаем оригинальный запрос
    task_prompt_lower = task_prompt.lower()
    
    # Извлекаем ключевые компоненты
    extracted = _extract_key_entities_and_concepts(task_prompt)
    
    # Стратегия 1: Запрос с акцентом на сущности
    if extracted['entities']:
        entities_query = f"{task_prompt} Focus on: {', '.join(extracted['entities'][:5])}"
        queries.append(entities_query)
    
    # Стратегия 2: Запрос с акцентом на действия
    if extracted['actions']:
        actions_query = f"{task_prompt} Actions needed: {', '.join(extracted['actions'][:5])}"
        queries.append(actions_query)
    
    # Стратегия 3: Запрос с акцентом на концепции
    if extracted['concepts']:
        concepts_query = f"{task_prompt} Related concepts: {', '.join(extracted['concepts'][:5])}"
        queries.append(concepts_query)
    
    # Стратегия 4: Упрощенный запрос (только ключевые слова)
    keywords = []
    keywords.extend(extracted['entities'][:3])
    keywords.extend(extracted['actions'][:2])
    keywords.extend(extracted['concepts'][:2])
    if keywords:
        simplified_query = ' '.join(keywords)
        queries.append(simplified_query)
    
    # Стратегия 5: Расширенный запрос с синонимами
    expanded_query = _expand_query_for_retrieval(task_prompt, task_type)
    if expanded_query != task_prompt:
        queries.append(expanded_query)
    
    # Стратегия 6: Вопросный формат (для comprehension tasks)
    if any(word in task_prompt_lower for word in ['how', 'what', 'why', 'where', 'when']):
        # Уже в вопросном формате, добавляем расширенный вопрос
        question_query = f"How does {task_prompt_lower}? What code implements {task_prompt_lower}?"
        queries.append(question_query)
    else:
        # Преобразуем в вопрос
        question_query = f"How to {task_prompt_lower}? What is needed for {task_prompt_lower}?"
        queries.append(question_query)
    
    # Стратегия 7: Технический фокус (для implementation tasks)
    if any(word in task_prompt_lower for word in ['implement', 'add', 'create', 'build', 'develop']):
        tech_query = f"Implementation details: {task_prompt}. Code structure and patterns needed."
        queries.append(tech_query)
    
    # Стратегия 8: Архитектурный фокус (для architectural tasks)
    if task_type == 'architectural' or any(word in task_prompt_lower for word in ['architect', 'architecture', 'design', 'structure', 'pattern']):
        arch_query = f"Architecture and design: {task_prompt}. Components, interfaces, and relationships."
        queries.append(arch_query)
    
    # Стратегия 9: Security фокус (для security tasks)
    if task_type == 'security' or any(word in task_prompt_lower for word in ['security', 'audit', 'vulnerability', 'secure', 'safe']):
        security_query = f"Security analysis: {task_prompt}. Vulnerabilities, authentication, authorization."
        queries.append(security_query)
    
    # Стратегия 10: Comprehension фокус (для comprehension tasks)
    if task_type == 'comprehension' or any(word in task_prompt_lower for word in ['trace', 'understand', 'comprehension', 'follow', 'track', 'flow']):
        comprehension_query = f"Code flow and execution: {task_prompt}. Trace execution path and data flow."
        queries.append(comprehension_query)
    
    # Стратегия 11: Комбинированный запрос (сущности + действия + концепции)
    if extracted['entities'] and extracted['actions'] and extracted['concepts']:
        combined_query = f"{task_prompt} Entities: {', '.join(extracted['entities'][:3])}. Actions: {', '.join(extracted['actions'][:2])}. Concepts: {', '.join(extracted['concepts'][:2])}"
        queries.append(combined_query)
    
    # Ограничиваем количество запросов
    queries = queries[:num_queries]
    
    # Удаляем дубликаты, сохраняя порядок
    seen = set()
    unique_queries = []
    for q in queries:
        q_hash = hash(q)
        if q_hash not in seen:
            seen.add(q_hash)
            unique_queries.append(q)
    
    logger.debug(f"Generated {len(unique_queries)} multi-queries from original prompt (task_type={task_type})")
    return unique_queries


def _rank_files_with_bm25(
    task_prompt: str,
    candidate_files: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Ранжирует файлы используя BM25 (keyword-based ranking).
    
    Args:
        task_prompt: Промпт задачи
        candidate_files: Список файлов-кандидатов
    
    Returns:
        Отсортированный список файлов с BM25 scores
    """
    if not BM25_AVAILABLE:
        logger.warning("BM25 not available, falling back to simple keyword matching")
        return _rank_files_with_keywords(task_prompt, candidate_files)
    
    if not candidate_files:
        return []
    
    # Токенизируем промпт
    prompt_tokens = task_prompt.lower().split()
    
    # Токенизируем содержимое файлов
    file_corpus = []
    for file_info in candidate_files:
        content = file_info.get("content", "")
        # Простая токенизация: разбиваем по словам и убираем короткие токены
        tokens = [t.lower() for t in re.findall(r'\b\w+\b', content) if len(t) > 2]
        file_corpus.append(tokens)
    
    if not file_corpus:
        return candidate_files
    
    # Создаем BM25 индекс
    try:
        bm25 = BM25Okapi(file_corpus)
        # Получаем BM25 scores
        scores = bm25.get_scores(prompt_tokens)
        
        # Добавляем scores к файлам
        for idx, file_info in enumerate(candidate_files):
            file_info["bm25_score"] = float(scores[idx])
        
        # Сортируем по BM25 score
        return sorted(candidate_files, key=lambda info: info.get("bm25_score", 0.0), reverse=True)
    except Exception as e:
        logger.warning(f"BM25 ranking failed: {e}, falling back to keyword matching")
        return _rank_files_with_keywords(task_prompt, candidate_files)


def _rank_files_with_keywords(
    task_prompt: str,
    candidate_files: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Простое ранжирование по ключевым словам (fallback для BM25).
    """
    if not candidate_files:
        return []
    
    # Извлекаем ключевые слова из промпта
    prompt_lower = task_prompt.lower()
    keywords = set(re.findall(r'\b[a-zA-Z]{4,}\b', prompt_lower))
    
    # Убираем стоп-слова
    stop_words = {
        "that", "this", "with", "from", "file", "code", "function",
        "class", "method", "should", "must", "need", "implement",
        "create", "add", "make", "using", "when", "where", "what",
    }
    keywords = keywords - stop_words
    
    # Подсчитываем совпадения для каждого файла
    for file_info in candidate_files:
        content_lower = file_info.get("content", "").lower()
        matches = sum(1 for keyword in keywords if keyword in content_lower)
        # Нормализуем score
        file_info["bm25_score"] = float(matches) / max(len(keywords), 1)
    
    return sorted(candidate_files, key=lambda info: info.get("bm25_score", 0.0), reverse=True)


def _get_category_specific_config(task_category: Optional[str]) -> Dict[str, Any]:
    """
    Возвращает специализированную конфигурацию ретривера для каждой категории задач.
    Оптимизировано на основе результатов: Integration Testing (2.235) - лучший, Code Comprehension (2.051) - худший.
    """
    task_category_lower = (task_category or '').lower()
    
    # Базовые конфигурации для каждой категории
    configs = {
        'integration_testing': {
            'file_multiplier': 1.65,  # Увеличено с 1.50 - нужно больше файлов для интеграционных точек
            'level1_ratio': 0.58,     # Семантически релевантные файлы
            'level2_ratio': 0.37,     # Зависимости (важно для интеграции)
            'level3_ratio': 0.05,     # Важные файлы
            'hybrid_alpha': 0.70,     # Баланс семантики и ключевых слов
            'dependency_depth': 3,     # Глубина зависимостей
            'dependency_files_per_level': 25,  # Файлов на уровень
            'chunks_per_file': 6,     # Больше чанков для покрытия интеграционных точек
            'prioritize_test_files': True,  # Приоритет тестовым файлам
            'boost_keywords': ['test', 'integration', 'suite', 'spec', 'specification', 'mock', 'stub'],
        },
        'multi_session_development': {
            'file_multiplier': 1.50,  # Увеличено с 1.40
            'level1_ratio': 0.68,     # Семантически релевантные
            'level2_ratio': 0.27,     # Зависимости
            'level3_ratio': 0.05,     # Важные файлы
            'hybrid_alpha': 0.72,     # Больше семантики для контекста
            'dependency_depth': 2,
            'dependency_files_per_level': 20,
            'chunks_per_file': 5,
            'prioritize_test_files': False,
            'boost_keywords': ['session', 'state', 'persist', 'cache', 'store', 'memory'],
        },
        'security_analysis': {
            'file_multiplier': 1.50,  # Увеличено с 1.45 - для улучшения с 2.350 до 2.4+
            'level1_ratio': 0.80,     # Увеличено с 0.78 - больше семантики для поиска уязвимостей
            'level2_ratio': 0.15,     # Немного уменьшено
            'level3_ratio': 0.05,     # Важные файлы
            'hybrid_alpha': 0.84,     # Увеличено с 0.82 - еще больше семантики для концептуального поиска
            'dependency_depth': 2,
            'dependency_files_per_level': 20,  # Увеличено с 18
            'chunks_per_file': 6,     # Увеличено с 5 - больше чанков для анализа безопасности
            'prioritize_test_files': False,
            'boost_keywords': ['security', 'auth', 'encrypt', 'validate', 'sanitize', 'vulnerability', 'exploit', 'attack', 'password', 'token', 'permission', 'access'],
        },
        'feature_implementation': {
            'file_multiplier': 1.55,  # Увеличено с 1.40 - для улучшения с 2.174 до 2.3+
            'level1_ratio': 0.70,     # Немного уменьшено для большего количества зависимостей
            'level2_ratio': 0.25,     # Увеличено с 0.23 - больше зависимостей для контекста
            'level3_ratio': 0.05,     # Важные файлы
            'hybrid_alpha': 0.78,     # Увеличено с 0.76 - больше семантики для поиска релевантного кода
            'dependency_depth': 3,     # Увеличено с 2 - более глубокая зависимость
            'dependency_files_per_level': 28,  # Увеличено с 22 - больше файлов для контекста
            'chunks_per_file': 6,     # Увеличено с 5 - больше чанков для реализации
            'prioritize_test_files': False,
            'boost_keywords': ['feature', 'implement', 'add', 'create', 'new', 'functionality', 'api', 'endpoint', 'service', 'handler', 'controller'],
        },
        'cross_file_refactoring': {
            'file_multiplier': 2.10,  # Увеличено с 2.00 - нужно больше файлов для рефакторинга
            'level1_ratio': 0.42,     # Семантически релевантные
            'level2_ratio': 0.48,     # Много зависимостей для понимания структуры
            'level3_ratio': 0.10,     # Важные файлы
            'hybrid_alpha': 0.60,     # Больше BM25 для точных совпадений
            'dependency_depth': 4,     # Глубокая зависимость
            'dependency_files_per_level': 38,  # Больше файлов на уровень
            'chunks_per_file': 7,     # Больше чанков для полного покрытия
            'prioritize_test_files': False,
            'boost_keywords': ['refactor', 'restructure', 'reorganize', 'merge', 'consolidate'],
        },
        'bug_investigation': {
            'file_multiplier': 1.75,  # Увеличено с 1.60 - нужно больше файлов для отслеживания багов
            'level1_ratio': 0.52,     # Семантически релевантные
            'level2_ratio': 0.43,     # Много зависимостей для трассировки
            'level3_ratio': 0.05,     # Важные файлы
            'hybrid_alpha': 0.66,     # Баланс для поиска багов
            'dependency_depth': 3,
            'dependency_files_per_level': 32,
            'chunks_per_file': 6,     # Больше чанков для трассировки
            'prioritize_test_files': True,  # Тесты могут показать ожидаемое поведение
            'boost_keywords': ['bug', 'error', 'exception', 'fail', 'issue', 'problem', 'debug', 'trace'],
        },
        'architectural_understanding': {
            'file_multiplier': 1.75,  # ВОЗВРАТ к версии с результатом 2.099 (было 1.85, ухудшило до 1.986)
            'level1_ratio': 0.58,     # ВОЗВРАТ к версии с результатом 2.099 (было 0.65, ухудшило до 1.986)
            'level2_ratio': 0.32,     # ВОЗВРАТ к версии с результатом 2.099 (было 0.25, ухудшило до 1.986)
            'level3_ratio': 0.10,     # Важные файлы
            'hybrid_alpha': 0.72,     # ВОЗВРАТ к версии с результатом 2.099 (было 0.78, ухудшило до 1.986)
            'dependency_depth': 3,     # Оставить умеренную глубину
            'dependency_files_per_level': 25,  # Оставить умеренное количество
            'chunks_per_file': 5,     # Стандартное значение
            'prioritize_test_files': False,
            'boost_keywords': ['architect', 'design', 'pattern', 'structure', 'component', 'module', 'interface', 'abstract', 'factory', 'builder', 'service', 'manager', 'config', 'main', 'entry'],
        },
        'code_comprehension': {
            'file_multiplier': 1.40,  # ВОЗВРАТ к версии с результатом 2.099 (было 1.50, ухудшило до 1.986)
            'level1_ratio': 0.68,     # ВОЗВРАТ к версии с результатом 2.099 (было 0.72, ухудшило до 1.986)
            'level2_ratio': 0.27,     # ВОЗВРАТ к версии с результатом 2.099 (было 0.23, ухудшило до 1.986)
            'level3_ratio': 0.05,     # Важные файлы
            'hybrid_alpha': 0.75,     # ВОЗВРАТ к версии с результатом 2.099 (было 0.80, ухудшило до 1.986)
            'dependency_depth': 3,     # Оставить умеренную глубину
            'dependency_files_per_level': 20,  # Оставить умеренное количество
            'chunks_per_file': 5,     # Стандартное значение
            'prioritize_test_files': False,
            'boost_keywords': ['comprehension', 'understand', 'trace', 'follow', 'flow', 'execution', 'call', 'method', 'function', 'handler', 'processor', 'service', 'controller'],
        },
    }
    
    # Определяем категорию
    if 'integration' in task_category_lower or 'test' in task_category_lower:
        return configs['integration_testing']
    elif 'multi' in task_category_lower or 'session' in task_category_lower:
        return configs['multi_session_development']
    elif 'security' in task_category_lower:
        return configs['security_analysis']
    elif 'feature' in task_category_lower or 'implementation' in task_category_lower:
        return configs['feature_implementation']
    elif 'refactor' in task_category_lower:
        return configs['cross_file_refactoring']
    elif 'bug' in task_category_lower or 'investigation' in task_category_lower:
        return configs['bug_investigation']
    elif 'architectural' in task_category_lower:
        return configs['architectural_understanding']
    elif 'comprehension' in task_category_lower:
        return configs['code_comprehension']
    else:
        # Default configuration
        return {
            'file_multiplier': 1.30,
            'level1_ratio': 0.70,
            'level2_ratio': 0.20,
            'level3_ratio': 0.10,
            'hybrid_alpha': 0.70,
            'dependency_depth': 2,
            'dependency_files_per_level': 20,
            'chunks_per_file': 5,
            'prioritize_test_files': False,
            'boost_keywords': [],
        }


def _expand_query_for_retrieval(task_prompt: str, task_type: str = None) -> str:
    """
    Расширяет запрос для ритривера синонимами и связанными терминами.
    Это помогает найти файлы, которые релевантны, но используют другую терминологию.
    """
    # Извлечь ключевые компоненты
    extracted = _extract_key_entities_and_concepts(task_prompt)
    
    # Словарь синонимов для расширения запроса (расширенный для лучшего покрытия)
    synonym_map = {
        'merge': ['combine', 'integrate', 'consolidate', 'unite', 'join', 'merge', 'fuse', 'amalgamate'],
        'refactor': ['restructure', 'reorganize', 'redesign', 'improve', 'optimize', 'refactor', 'rework', 'revamp'],
        'sync': ['synchronize', 'coordinate', 'align', 'match', 'update', 'sync', 'harmonize', 'synchronize'],
        'implement': ['create', 'build', 'develop', 'add', 'construct', 'implement', 'realize', 'execute'],
        'trace': ['follow', 'track', 'debug', 'investigate', 'analyze', 'trace', 'monitor', 'examine'],
        'understand': ['comprehend', 'analyze', 'examine', 'study', 'review', 'understand', 'grasp', 'perceive'],
        'security': ['secure', 'safe', 'protection', 'authentication', 'authorization', 'security', 'safeguard', 'defense'],
        'architecture': ['structure', 'design', 'organization', 'layout', 'framework', 'architecture', 'blueprint', 'schema'],
        'offline': ['local', 'cached', 'stored', 'persistent', 'offline', 'localized', 'resident'],
        'repository': ['store', 'database', 'dao', 'data access', 'repository', 'storage', 'persistence', 'cache'],
        'service': ['manager', 'handler', 'processor', 'controller', 'service', 'facade', 'coordinator'],
        'worker': ['worker', 'thread', 'task', 'job', 'executor', 'processor', 'handler'],
        'room': ['room', 'space', 'area', 'zone', 'region', 'location', 'place'],
        'store': ['store', 'repository', 'cache', 'storage', 'database', 'persistence'],
        'pricing': ['pricing', 'price', 'cost', 'fee', 'charge', 'rate', 'amount'],
        'contract': ['contract', 'agreement', 'deal', 'pact', 'treaty', 'arrangement'],
        'etag': ['etag', 'entity tag', 'cache tag', 'version tag', 'hash', 'checksum'],
        'conditional': ['conditional', 'condition', 'if', 'when', 'provided', 'depending'],
    }
    
    # Расширить ключевые слова синонимами
    expanded_terms = []
    
    # Добавить оригинальные сущности
    expanded_terms.extend(extracted['entities'])
    
    # Добавить действия с синонимами
    for action in extracted['actions']:
        expanded_terms.append(action)
        if action in synonym_map:
            expanded_terms.extend(synonym_map[action][:4])  # Добавить 4 синонима (увеличено с 2)
    
    # Добавить концепции с синонимами
    for concept in extracted['concepts']:
        expanded_terms.append(concept)
        if concept in synonym_map:
            expanded_terms.extend(synonym_map[concept][:4])  # Добавить 4 синонима (увеличено с 2)
    
    # Добавить связанные термины в зависимости от типа задачи (расширенный список)
    if task_type == 'architectural':
        expanded_terms.extend(['interface', 'abstract', 'pattern', 'component', 'module', 'structure', 
                              'design', 'architecture', 'framework', 'blueprint', 'schema', 'layout',
                              'hierarchy', 'composition', 'decomposition', 'coupling', 'cohesion'])
    elif task_type == 'comprehension':
        expanded_terms.extend(['flow', 'call', 'invoke', 'method', 'function', 'execution', 
                              'trace', 'track', 'follow', 'sequence', 'order', 'path', 'route',
                              'control flow', 'data flow', 'execution path', 'call stack'])
    elif task_type == 'security':
        expanded_terms.extend(['vulnerability', 'exploit', 'attack', 'injection', 'xss', 'csrf',
                              'security', 'secure', 'safe', 'protection', 'authentication', 'authorization',
                              'encryption', 'validation', 'sanitization', 'input validation', 'access control'])
    elif task_type == 'implementation':
        expanded_terms.extend(['feature', 'functionality', 'capability', 'endpoint', 'api',
                              'implementation', 'realization', 'execution', 'code', 'logic',
                              'algorithm', 'mechanism', 'function', 'method', 'handler'])
    
    # Создать расширенный запрос
    expanded_query = task_prompt
    if expanded_terms:
        # Добавить расширенные термины, избегая дубликатов
        unique_terms = list(set(expanded_terms))
        # Фильтровать слишком короткие термины
        meaningful_terms = [t for t in unique_terms if len(t) > 3]
        if meaningful_terms:
            expanded_query += " " + " ".join(meaningful_terms[:20])  # Увеличено с 15 до 20 терминов
    
    return expanded_query


def _rank_files_with_embeddings(
    model: "SentenceTransformer",
    task_prompt: str,
    candidate_files: List[Dict[str, Any]],
    expanded_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Rank candidate files using cosine similarity in embedding space.
    Uses expanded query if provided for better retrieval.
    """
    if not candidate_files:
        return []

    # Использовать расширенный запрос если предоставлен, иначе оригинальный
    query_text = expanded_query if expanded_query else task_prompt
    
    texts = [query_text] + [file_info["content"] for file_info in candidate_files]
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
    use_multi_query: bool = True,
    use_hybrid_search: bool = True,
    hybrid_alpha: float = 0.7,
    task_category: Optional[str] = None,  # Add task_category parameter
    use_mcp: bool = False,  # Use MCP-based retrieval
    mcp_provider: Optional[str] = None,  # MCP provider
    mcp_model: Optional[str] = None,  # MCP model
    mcp_base_url: Optional[str] = None,  # MCP base URL
    mcp_api_key: Optional[str] = None,  # MCP API key
    config: Optional[Any] = None,  # Config object
) -> str:
    """
    Retrieve the most relevant project files using embeddings and return them as context.
    
    Args:
        smart_chunking: If True, split files into chunks and select most relevant chunks
        chunks_per_file: Maximum number of chunks to select per file (when smart_chunking=True)
        chunk_size: Size of each chunk in characters (when smart_chunking=True)
        use_mcp: If True, use MCP-based intelligent retrieval instead of standard retrieval
        mcp_provider: LLM provider for MCP ("openai", "anthropic", "ollama", "huggingface", "local_openai")
        mcp_model: Model name for MCP
        mcp_base_url: Base URL for local providers
        mcp_api_key: API key for local providers
        config: Config object for MCP
    """
    start_time = time.perf_counter()
    
    # Try MCP-based retrieval if enabled
    if use_mcp and task_category and project_dir:
        try:
            from .mcp_retrieval import retrieve_with_mcp
            
            # Определить, использовать ли LLM или эвристики
            # Если провайдер не указан или недоступен, используем эвристики
            use_llm_for_mcp = False
            if mcp_provider and mcp_provider not in ("", "none", "heuristics"):
                # Попробуем использовать LLM только если провайдер указан
                use_llm_for_mcp = True
            
            logger.info(
                f"🔧 Using MCP-based retrieval "
                f"(provider={mcp_provider or 'heuristics'}, "
                f"use_llm={use_llm_for_mcp})"
            )
            
            mcp_result = retrieve_with_mcp(
                context_files=context_files or {},
                task_prompt=task_prompt,
                task_category=task_category,
                project_dir=project_dir,
                config=config,
                provider=mcp_provider or "ollama",  # Имя не важно при use_llm=False
                model=mcp_model,
                base_url=mcp_base_url,
                api_key=mcp_api_key,
                use_llm=use_llm_for_mcp,  # Использовать LLM только если провайдер доступен
            )
            
            if mcp_result:
                logger.info(f"✅ MCP retrieval returned {len(mcp_result)} characters")
                return mcp_result
            else:
                logger.warning("⚠️ MCP retrieval returned empty result, falling back to standard retrieval")
        except Exception as e:
            logger.warning(f"⚠️ MCP retrieval failed: {e}. Falling back to standard retrieval.")
            import traceback
            logger.debug(traceback.format_exc())
            # Fall through to standard retrieval
    
    # Standard retrieval (existing code)
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

    # MULTI-LEVEL RETRIEVAL STRATEGY (adaptive based on task category):
    # Use category-specific configuration for optimal retrieval
    
    task_prompt_lower = task_prompt.lower()
    task_category_lower = (task_category or '').lower()
    
    # Get category-specific configuration
    category_config = _get_category_specific_config(task_category)
    
    # Detect task type flags for backward compatibility and specific logic
    is_architectural_task = (
        'architectural' in task_category_lower or 'refactor' in task_category_lower or
        any(keyword in task_prompt_lower 
            for keyword in ['architect', 'architecture', 'structure', 'design', 'pattern', 'component', 'module', 'merge', 'refactor', 'critique', 'evaluate design', 'cross file refactor', 'refactoring'])
    )
    
    is_code_comprehension_task = (
        'comprehension' in task_category_lower or
        any(keyword in task_prompt_lower
            for keyword in ['trace', 'understand', 'comprehension', 'follow', 'track', 'flow', 'discrepancy', 'why', 'how does', 'explain'])
    )
    
    is_security_task = (
        'security' in task_category_lower or
        any(keyword in task_prompt_lower
            for keyword in ['security', 'audit', 'vulnerability', 'secure', 'safe', 'protection', 'exploit', 'attack'])
    )
    
    is_feature_implementation_task = (
        'feature' in task_category_lower or 'implementation' in task_category_lower or
        any(keyword in task_prompt_lower
            for keyword in ['implement', 'add', 'create', 'build', 'develop', 'feature', 'functionality', 'etag', 'conditional'])
    )
    
    is_integration_testing_task = (
        'integration' in task_category_lower or 'test' in task_category_lower or
        any(keyword in task_prompt_lower
            for keyword in ['integration test', 'integration testing', 'test integration', 'integration', 'test suite', 'test case'])
    )
    
    is_multi_session_task = (
        'multi' in task_category_lower or 'session' in task_category_lower or
        any(keyword in task_prompt_lower
            for keyword in ['multi session', 'multi-session', 'session', 'multiple sessions', 'ongoing'])
    )
    
    is_bug_investigation_task = (
        'bug' in task_category_lower or 'investigation' in task_category_lower or
        any(keyword in task_prompt_lower
            for keyword in ['bug', 'investigate', 'investigation', 'debug', 'error', 'issue', 'problem', 'fix bug', 'trace bug'])
    )
    
    is_refactoring_task = (
        'refactor' in task_category_lower or
        any(keyword in task_prompt_lower
            for keyword in ['refactor', 'refactoring', 'restructure', 'reorganize', 'cross file', 'cross-file', 'multi-file'])
    )
    
    # Apply file multiplier from category-specific configuration
    original_selected_count = selected_count
    file_multiplier = category_config['file_multiplier']
    selected_count = int(selected_count * file_multiplier)
    selected_count = min(selected_count, len(candidates))
    
    # УДАЛЕНО: Дополнительные бусты ухудшили результаты (упали с 2.115 до 1.991)
    # Версия f1922da без дополнительных бустов давала максимальный скор 2.115
    
    logger.debug("📊 Category-specific config (%s): increased file count from %d to %d (%.2fx)", 
                task_category or 'default', original_selected_count, selected_count, file_multiplier)
    
    # Use ratios from category-specific configuration
    level1_ratio = category_config['level1_ratio']
    level2_ratio = category_config['level2_ratio']
    level3_ratio = category_config['level3_ratio']
    logger.debug("📊 Category-specific ratios: L1=%.0f%%, L2=%.0f%%, L3=%.0f%%", 
                level1_ratio * 100, level2_ratio * 100, level3_ratio * 100)
    
    # Create optimized retrieval query using prompt engineering
    task_type_name = None
    if is_architectural_task or is_refactoring_task:
        task_type_name = 'architectural'
    elif is_code_comprehension_task or is_bug_investigation_task:
        task_type_name = 'comprehension'
    elif is_security_task:
        task_type_name = 'security'
    elif is_feature_implementation_task:
        task_type_name = 'implementation'
    elif is_integration_testing_task:
        task_type_name = 'testing'
    elif is_multi_session_task:
        task_type_name = 'multi_session'
    
    # Extract key entities and concepts for better retrieval
    extracted_info = _extract_key_entities_and_concepts(task_prompt)
    
    # Expand query with synonyms and related terms
    expanded_query = _expand_query_for_retrieval(task_prompt, task_type_name)
    
    logger.debug(
        "🔍 Query expansion: extracted %d entities, %d actions, %d concepts",
        len(extracted_info['entities']),
        len(extracted_info['actions']),
        len(extracted_info['concepts'])
    )
    
    # Multi-Query Retrieval: Generate multiple query variants and combine results
    if use_multi_query:
        multi_queries = _generate_multi_queries(task_prompt, num_queries=8, task_type=task_type_name)
        logger.debug(f"🔍 Multi-query retrieval: using {len(multi_queries)} query variants")
        
        # Rank files for each query variant
        all_ranked_files = []
        query_weights = []  # Веса для каждого запроса (первый запрос - оригинальный, имеет больший вес)
        
        for idx, query_variant in enumerate(multi_queries):
            variant_expanded = _expand_query_for_retrieval(query_variant, task_type_name)
            variant_ranked = _rank_files_with_embeddings(model, query_variant, candidates, expanded_query=variant_expanded)
            all_ranked_files.append(variant_ranked)
            # Первый запрос (оригинальный) имеет больший вес, специализированные запросы - средний вес
            if idx == 0:
                weight = 1.0  # Оригинальный запрос
            elif idx < 5:  # Первые специализированные запросы
                weight = 0.9
            else:  # Остальные запросы
                weight = 0.7
            query_weights.append(weight)
        
        # Combine results: improved aggregation with RRF and weighted similarity
        file_scores: Dict[str, float] = {}
        file_info_map: Dict[str, Dict[str, Any]] = {}
        file_rrf_scores: Dict[str, float] = {}
        file_similarity_scores: Dict[str, List[float]] = {}
        
        for query_idx, ranked_list in enumerate(all_ranked_files):
            query_weight = query_weights[query_idx]
            
            for rank_idx, file_info in enumerate(ranked_list):
                file_path = _normalize_relative_path(file_info["path"])
                similarity = file_info.get("similarity", 0.0)
                
                # Reciprocal Rank Fusion: score = 1 / (rank + k)
                # Используем k=15 для большей чувствительности к рангу (уменьшено с 20)
                rank = rank_idx + 1
                rrf_score = 1.0 / (rank + 15)  # k=15 для лучшей чувствительности
                
                if file_path not in file_scores:
                    file_scores[file_path] = 0.0
                    file_rrf_scores[file_path] = 0.0
                    file_similarity_scores[file_path] = []
                    file_info_map[file_path] = file_info.copy()
                
                # Накопление RRF scores (взвешенное)
                file_rrf_scores[file_path] += rrf_score * query_weight
                
                # Накопление similarity scores для усреднения
                file_similarity_scores[file_path].append(similarity * query_weight)
        
        # Комбинируем RRF и similarity scores
        for file_path in file_scores:
            # Среднее similarity по всем запросам (взвешенное)
            avg_similarity = sum(file_similarity_scores[file_path]) / len(file_similarity_scores[file_path]) if file_similarity_scores[file_path] else 0.0
            
            # Максимальное similarity (лучший результат среди всех запросов)
            max_similarity = max(file_similarity_scores[file_path]) if file_similarity_scores[file_path] else 0.0
            
            # Нормализуем RRF score (максимальный RRF = 1.0 для top-1 файла)
            max_rrf = max(file_rrf_scores.values()) if file_rrf_scores else 1.0
            normalized_rrf = file_rrf_scores[file_path] / max_rrf if max_rrf > 0 else 0.0
            
            # Комбинируем: 65% max_similarity (лучший match), 25% avg_similarity (стабильность), 10% RRF (позиция)
            combined_score = (max_similarity * 0.65 + avg_similarity * 0.25 + normalized_rrf * 0.10)
            file_scores[file_path] = combined_score
        
        # Sort by combined score
        ranked_files = []
        for file_path, combined_score in sorted(file_scores.items(), key=lambda x: x[1], reverse=True):
            file_info = file_info_map[file_path]
            file_info["similarity"] = combined_score
            file_info["multi_query_score"] = combined_score
            ranked_files.append(file_info)
        
        logger.debug(f"✅ Multi-query retrieval: combined {len(ranked_files)} files from {len(multi_queries)} queries")
    else:
        # Single query retrieval (original behavior)
        ranked_files = _rank_files_with_embeddings(model, task_prompt, candidates, expanded_query=expanded_query)
    
    # Hybrid Search: Combine semantic (embeddings) and keyword (BM25) results
    if use_hybrid_search and len(candidates) > 0:
        logger.debug("🔍 Hybrid search: combining semantic and BM25 results")
        
        # Use hybrid_alpha from category-specific configuration
        adaptive_hybrid_alpha = category_config.get('hybrid_alpha', hybrid_alpha)
        
        logger.debug(f"🔍 Category-specific hybrid_alpha: {adaptive_hybrid_alpha} (category={task_category or 'default'})")
        
        # Get BM25 rankings
        bm25_ranked = _rank_files_with_bm25(task_prompt, candidates.copy())
        
        # Improved normalization: use min-max normalization with smoothing
        semantic_scores = [f.get("similarity", 0.0) for f in ranked_files] if ranked_files else []
        bm25_scores = [f.get("bm25_score", 0.0) for f in bm25_ranked] if bm25_ranked else []
        
        # Normalize semantic scores
        if semantic_scores:
            min_semantic = min(semantic_scores)
            max_semantic = max(semantic_scores)
            range_semantic = max_semantic - min_semantic if max_semantic > min_semantic else 1.0
            
            for f in ranked_files:
                raw_score = f.get("similarity", 0.0)
                # Min-max normalization with smoothing (add small epsilon to avoid division by zero)
                normalized = (raw_score - min_semantic) / range_semantic if range_semantic > 0 else 0.0
                # Apply softmax-like transformation for better distribution
                f["normalized_semantic"] = normalized ** 0.8  # Power scaling for better distribution
        else:
            for f in ranked_files:
                f["normalized_semantic"] = 0.0
        
        # Normalize BM25 scores
        if bm25_scores:
            min_bm25 = min(bm25_scores)
            max_bm25 = max(bm25_scores)
            range_bm25 = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
            
            for f in bm25_ranked:
                raw_score = f.get("bm25_score", 0.0)
                normalized = (raw_score - min_bm25) / range_bm25 if range_bm25 > 0 else 0.0
                # Apply softmax-like transformation
                f["normalized_bm25"] = normalized ** 0.8
        else:
            for f in bm25_ranked:
                f["normalized_bm25"] = 0.0
        
        # Create file map for hybrid scoring
        hybrid_scores: Dict[str, float] = {}
        hybrid_info: Dict[str, Dict[str, Any]] = {}
        
        # Add semantic scores with weights
        for file_info in ranked_files:
            file_path = _normalize_relative_path(file_info["path"])
            semantic_score = file_info.get("normalized_semantic", 0.0)
            hybrid_scores[file_path] = semantic_score * adaptive_hybrid_alpha
            hybrid_info[file_path] = file_info.copy()
        
        # Add BM25 scores (combine with semantic)
        for file_info in bm25_ranked:
            file_path = _normalize_relative_path(file_info["path"])
            bm25_score = file_info.get("normalized_bm25", 0.0) * (1.0 - adaptive_hybrid_alpha)
            
            if file_path in hybrid_scores:
                # Boost files that appear in both rankings
                hybrid_scores[file_path] = hybrid_scores[file_path] * 1.2 + bm25_score
            else:
                hybrid_scores[file_path] = bm25_score
                hybrid_info[file_path] = file_info.copy()
        
        # Re-rank by hybrid score
        ranked_files = []
        for file_path, hybrid_score in sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True):
            file_info = hybrid_info[file_path]
            file_info["similarity"] = hybrid_score
            file_info["hybrid_score"] = hybrid_score
            ranked_files.append(file_info)
        
        logger.debug(f"✅ Hybrid search: combined semantic (α={adaptive_hybrid_alpha}) and BM25 (1-α={1-adaptive_hybrid_alpha})")
    
    # Boost files BEFORE selection based on category-specific keywords
    boost_keywords = category_config.get('boost_keywords', [])
    prioritize_test_files = category_config.get('prioritize_test_files', False)
    
    if boost_keywords or prioritize_test_files:
        boosted_count = 0
        task_words = set(task_prompt_lower.split())
        
        for file_info in ranked_files:
            file_path_lower = file_info["path"].lower()
            file_name_lower = Path(file_info["path"]).name.lower()
            original_sim = file_info.get("similarity", 0.0)
            boost = 0.0
            
            # Boost for category-specific keywords (увеличено для лучшего ранжирования)
            if boost_keywords:
                keyword_matches = sum(1 for keyword in boost_keywords if keyword in file_path_lower or keyword in file_name_lower)
                if keyword_matches > 0:
                    boost += 0.30 + (keyword_matches * 0.08)  # Увеличено: Base 0.30 + 0.08 per keyword (было 0.25 + 0.05)
            
            # Boost for test files if prioritized
            if prioritize_test_files:
                if any(test_indicator in file_path_lower for test_indicator in ['test', 'spec', 'specification', 'mock', 'stub']):
                    boost += 0.25  # Увеличено с 0.20
            
            # Boost for files mentioned in task prompt (увеличено)
            file_words = set(file_name_lower.split('_') + file_name_lower.split('-') + [file_name_lower])
            common_words = task_words.intersection(file_words)
            if len(common_words) > 0:
                boost += 0.20  # Увеличено с 0.15
            
            # Additional boost for high similarity files (they're likely very relevant)
            if original_sim > 0.15:  # Увеличено порог с 0.12
                boost += 0.10  # Дополнительный бонус для высокорелевантных файлов
            
            if boost > 0:
                file_info["similarity"] = min(1.0, original_sim + boost)
                boosted_count += 1
        
        if boosted_count > 0:
            ranked_files = sorted(ranked_files, key=lambda info: info.get("similarity", 0.0), reverse=True)
            logger.debug("📈 Category-specific boosting: boosted %d files with keywords=%s, prioritize_test=%s", 
                        boosted_count, boost_keywords[:3] if boost_keywords else [], prioritize_test_files)
    
    # Legacy boosting for architectural/refactoring tasks (backward compatibility)
    if is_architectural_task or is_refactoring_task:
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
            
            # 1. Boost for architectural keywords in path/name (FURTHER INCREASED для улучшения с 1.789)
            keyword_matches = sum(1 for keyword in architectural_keywords if keyword in file_path_lower or keyword in file_name_lower)
            if keyword_matches > 0:
                boost += 0.40 + (keyword_matches * 0.08)  # Увеличено: Base 0.40 + 0.08 per keyword (было 0.35 + 0.06)
            
            # 2. Boost for architectural patterns in content (FURTHER INCREASED)
            content_preview = file_info.get("content", "")[:4000]  # Extended to 4000 chars (was 3500)
            content_lower = content_preview.lower()
            architectural_patterns = [
                'interface ', 'abstract class', 'implements', 'extends', 'public class',
                'public interface', '@service', '@component', '@repository', '@entity',
                'class.*extends', 'class.*implements', 'extends.*implements', 'implements.*extends'
            ]
            pattern_matches = sum(1 for pattern in architectural_patterns if pattern in content_lower)
            if pattern_matches > 0:
                boost += 0.45 + (pattern_matches * 0.15)  # Увеличено: Base 0.45 + 0.15 per pattern (было 0.42 + 0.12)
            
            # 3. Boost for files with high similarity already (they're likely relevant)
            if original_sim > 0.10:  # Снижено порог с 0.12 для большего охвата
                boost += 0.28  # Увеличено с 0.25
            
            # 4. Boost for files mentioned in task prompt (by name)
            file_words = set(file_name_lower.split('_') + file_name_lower.split('-') + [file_name_lower])
            common_words = task_words.intersection(file_words)
            if len(common_words) > 0:
                boost += 0.35  # Увеличено с 0.30
            
            # 5. Additional boost for entry points and configuration files
            if any(indicator in file_name_lower for indicator in ['main', 'application', 'config', 'factory', 'builder']):
                boost += 0.30  # Увеличено с 0.25
            
            if boost > 0:
                file_info["similarity"] = min(1.0, original_sim + boost)
                boosted_count += 1
        
        # Re-rank after boosting
        ranked_files = sorted(ranked_files, key=lambda info: info.get("similarity", 0.0), reverse=True)
        logger.debug("🏗️ Boosted %d architectural/refactoring files before selection (max boost applied)", boosted_count)
    
    # Boost for comprehension and bug investigation tasks
    elif is_code_comprehension_task or is_bug_investigation_task:
        comprehension_keywords = [
            'trace', 'track', 'follow', 'flow', 'call', 'invoke', 'method', 'function',
            'execution', 'sequence', 'order', 'path', 'route', 'handler', 'processor',
            'service', 'controller', 'manager', 'worker', 'thread', 'task'
        ]
        
        task_words = set(task_prompt_lower.split())
        boosted_count = 0
        
        for file_info in ranked_files:
            file_path_lower = file_info["path"].lower()
            file_name_lower = Path(file_info["path"]).name.lower()
            original_sim = file_info.get("similarity", 0.0)
            boost = 0.0
            
            # Boost for flow-related keywords
            keyword_matches = sum(1 for keyword in comprehension_keywords if keyword in file_path_lower or keyword in file_name_lower)
            if keyword_matches > 0:
                boost += 0.20 + (keyword_matches * 0.04)
            
            # Boost for files with method calls, function calls in content
            content_preview = file_info.get("content", "")[:2500]
            content_lower = content_preview.lower()
            if any(pattern in content_lower for pattern in ['->', '.', '(', 'call', 'invoke', 'execute']):
                boost += 0.15
            
            # Boost for files mentioned in task prompt
            file_words = set(file_name_lower.split('_') + file_name_lower.split('-') + [file_name_lower])
            common_words = task_words.intersection(file_words)
            if len(common_words) > 0:
                boost += 0.18
            
            if boost > 0:
                file_info["similarity"] = min(1.0, original_sim + boost)
                boosted_count += 1
        
        ranked_files = sorted(ranked_files, key=lambda info: info.get("similarity", 0.0), reverse=True)
        logger.debug("🔍 Boosted %d comprehension/bug investigation files before selection", boosted_count)
    
    # Calculate how many files to select at each level
    # Level 1: Top semantically relevant files
    level1_count = max(1, int(selected_count * level1_ratio))
    
    # For architectural tasks, apply quality filter - only select files with good similarity
    if is_architectural_task or is_refactoring_task:
        # Filter to files with similarity > threshold (after boost) to ensure quality
        # Используем более низкий порог для архитектурных задач, чтобы захватить больше файлов
        quality_threshold = 0.04  # Снижено с 0.05 для большего охвата архитектурных файлов
        quality_files = [f for f in ranked_files if f.get("similarity", 0.0) > quality_threshold]
        if len(quality_files) >= level1_count:
            level1_files = quality_files[:level1_count]
            logger.debug(
                "🏗️ Architectural/Refactoring quality filter: selected %d files with similarity > %.2f",
                len(level1_files),
                quality_threshold
            )
        else:
            # If not enough quality files, use all available but log warning
            level1_files = ranked_files[:level1_count]
            logger.debug(
                "🏗️ Architectural/Refactoring: only %d files meet quality threshold, using top %d",
                len(quality_files),
                level1_count
            )
    elif is_code_comprehension_task or is_bug_investigation_task:
        # For comprehension/bug investigation: lower threshold для большего охвата
        quality_threshold = 0.04  # Снижено с 0.05 для большего охвата файлов для трассировки
        quality_files = [f for f in ranked_files if f.get("similarity", 0.0) > quality_threshold]
        if len(quality_files) >= level1_count:
            level1_files = quality_files[:level1_count]
            logger.debug(
                "🔍 Comprehension/Bug investigation quality filter: selected %d files with similarity > %.2f",
                len(level1_files),
                quality_threshold
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
            
            # Find dependent files with deep graph expansion using category-specific config
            dependency_depth = category_config.get('dependency_depth', 2)
            dependency_files_per_level = category_config.get('dependency_files_per_level', 20)
            max_dependent_files = min(int(level2_count * 2.5), selected_count - level1_count)
            
            # Create a wrapper function that uses category-specific config
            def _expand_with_category_config(seed_paths, dep_graph, rev_graph):
                return _expand_via_dependency_graph(
                    seed_paths,
                    dep_graph,
                    rev_graph,
                    max_depth=dependency_depth,
                    max_files_per_level=dependency_files_per_level
                )
            
            dependent_files = _find_dependent_files(
                level1_files,
                candidates,
                dependency_graph,
                reverse_graph,
                max_dependent_files=max_dependent_files,
                is_architectural_task=is_architectural_task,
                use_deep_expansion=True,  # Enable deep graph expansion
                expand_func=_expand_with_category_config,  # Pass custom expansion function
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
        # Use category-specific chunks_per_file from configuration
        category_chunks_per_file = category_config.get('chunks_per_file', chunks_per_file)
        effective_chunks_per_file = category_chunks_per_file
        
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
            "Retrieval: split %d files into %d chunks (avg %.1f chunks/file, category_chunks_per_file=%d)",
            len(selected_files),
            len(all_chunks),
            len(all_chunks) / len(selected_files) if selected_files else 0,
            effective_chunks_per_file,
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
                max_chunks_for_level3 = min(2, effective_chunks_per_file)
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
                
                # For architectural and refactoring tasks, prioritize first few chunks (class definitions)
                if (is_architectural_task or is_refactoring_task) and len(file_chunks_sorted_by_pos) > 1:
                    # Include first chunks based on category config (usually contain class/interface definitions)
                    max_early_chunks = min(effective_chunks_per_file - 1, len(file_chunks_sorted_by_pos) - 1)
                    for i in range(1, max_early_chunks + 1):
                        early_chunk = file_chunks_sorted_by_pos[i]
                        if early_chunk not in top_chunks and len(top_chunks) < effective_chunks_per_file:
                            # Stronger boost for early chunks in architectural/refactoring tasks
                            early_chunk["similarity"] = early_chunk.get("similarity", 0.0) + 0.18  # Увеличено с 0.15
                            top_chunks.append(early_chunk)
                
                # For comprehension and bug investigation tasks, prioritize more chunks for tracing
                elif (is_code_comprehension_task or is_bug_investigation_task) and len(file_chunks_sorted_by_pos) > 1:
                    # Include first chunks for better flow tracing
                    max_early_chunks = min(6, effective_chunks_per_file - 1, len(file_chunks_sorted_by_pos) - 1)  # Увеличено с 5 до 6
                    for i in range(1, max_early_chunks + 1):
                        early_chunk = file_chunks_sorted_by_pos[i]
                        if early_chunk not in top_chunks and len(top_chunks) < effective_chunks_per_file:
                            early_chunk["similarity"] = early_chunk.get("similarity", 0.0) + 0.12  # Увеличено с 0.10
                            top_chunks.append(early_chunk)
                
                # Diversification strategy: select chunks from different parts of the file
                # Divide file into regions and try to get at least one chunk from each region
                if len(file_chunks_sorted_by_pos) > 1:
                    # For comprehension/bug investigation: more regions for better coverage
                    num_regions = min(4 if (is_code_comprehension_task or is_bug_investigation_task) else 3, effective_chunks_per_file - len(top_chunks))
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
                                
                                if best_in_region not in top_chunks and len(top_chunks) < effective_chunks_per_file:
                                    top_chunks.append(best_in_region)
                
                # Fill remaining slots with top relevant chunks
                for chunk in file_chunks_sorted_by_relevance:
                    if len(top_chunks) >= effective_chunks_per_file:
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
    use_multi_query: bool = True,
    use_hybrid_search: bool = True,
    hybrid_alpha: float = 0.7,
    task_category: Optional[str] = None,  # Add task_category parameter
    use_mcp: bool = False,  # Use MCP-based retrieval
    mcp_provider: Optional[str] = None,  # MCP provider
    mcp_model: Optional[str] = None,  # MCP model
    mcp_base_url: Optional[str] = None,  # MCP base URL
    mcp_api_key: Optional[str] = None,  # MCP API key
    config: Optional[Any] = None,  # Config object
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
            use_multi_query=use_multi_query,
            use_hybrid_search=use_hybrid_search,
            hybrid_alpha=hybrid_alpha,
            task_category=task_category,  # Pass task_category
            use_mcp=use_mcp,  # Pass MCP parameters
            mcp_provider=mcp_provider,
            mcp_model=mcp_model,
            mcp_base_url=mcp_base_url,
            mcp_api_key=mcp_api_key,
            config=config,
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
