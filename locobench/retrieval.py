"""
Retrieval mechanism for LoCoBench

This module provides retrieval-augmented generation (RAG) capabilities
for hard and expert difficulty scenarios. It extracts relevant code fragments
from context files using embeddings-based similarity search.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Retrieval will fall back to keyword-based method.")


def split_code(code: str, chunk_size: int = 512) -> List[str]:
    """
    Split code into chunks for embedding-based retrieval.
    
    Args:
        code: Source code content
        chunk_size: Maximum chunk size in characters
        
    Returns:
        List of code chunks
    """
    if not code:
        return []
    
    lines = code.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line_size = len(line) + 1  # +1 for newline
        if current_size + line_size > chunk_size and current_chunk:
            # Save current chunk
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size
    
    # Add remaining chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks


def retrieve_relevant_embedding(
    context_files: Dict[str, str],
    task_prompt: str,
    top_k: int = 5,
    model_name: str = 'all-MiniLM-L6-v2'
) -> str:
    """
    Retrieve top-K relevant code fragments using embeddings.
    
    Args:
        context_files: Dictionary mapping file paths to code content
        task_prompt: The task description/prompt
        top_k: Number of top fragments to retrieve
        model_name: Name of the sentence transformer model
        
    Returns:
        Formatted string with retrieved code fragments
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("sentence-transformers not available. Cannot use embedding-based retrieval.")
        return ""
    
    if not context_files:
        logger.warning("No context files provided for retrieval")
        return ""
    
    try:
        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Prepare chunks with metadata
        chunks = []
        chunk_info = []  # (file_path, chunk_index, chunk_content)
        
        for file_path, code_content in context_files.items():
            file_chunks = split_code(code_content)
            for idx, chunk in enumerate(file_chunks):
                chunk_info.append((file_path, idx, chunk))
                chunks.append(chunk)
        
        if not chunks:
            logger.warning("No code chunks created from context files")
            return ""
        
        # Compute embeddings for chunks and query
        logger.info(f"Computing embeddings for {len(chunks)} chunks and query")
        all_texts = chunks + [task_prompt]
        embeddings = model.encode(all_texts, show_progress_bar=False)
        
        query_embedding = embeddings[-1]  # Last embedding is for task_prompt
        chunk_embeddings = embeddings[:-1]
        
        # Compute cosine similarity
        # Normalize embeddings
        query_norm = np.linalg.norm(query_embedding)
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
        
        # Avoid division by zero
        if query_norm == 0:
            logger.warning("Query embedding has zero norm")
            return ""
        
        # Compute cosine similarities
        similarities = np.dot(chunk_embeddings, query_embedding) / (chunk_norms * query_norm)
        
        # Handle potential NaN values
        similarities = np.nan_to_num(similarities, nan=0.0)
        
        # Get top-K indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]  # Sort descending
        
        # Format retrieved fragments
        retrieved_parts = []
        for idx in top_indices:
            file_path, chunk_idx, chunk_content = chunk_info[idx]
            similarity_score = similarities[idx]
            retrieved_parts.append(
                f"From {file_path} (chunk {chunk_idx + 1}, similarity: {similarity_score:.3f}):\n{chunk_content}"
            )
        
        retrieved_text = "\n\n".join(retrieved_parts)
        logger.info(f"Retrieved {len(top_indices)} fragments for retrieval")
        
        return retrieved_text
        
    except Exception as e:
        logger.error(f"Error during embedding-based retrieval: {e}", exc_info=True)
        return ""


def retrieve_relevant_keyword(
    context_files: Dict[str, str],
    task_prompt: str,
    top_k: int = 5
) -> str:
    """
    Retrieve relevant code fragments using keyword matching (simple fallback).
    
    Args:
        context_files: Dictionary mapping file paths to code content
        task_prompt: The task description/prompt
        top_k: Number of top fragments to retrieve
        
    Returns:
        Formatted string with retrieved code fragments
    """
    if not context_files:
        logger.warning("No context files provided for retrieval")
        return ""
    
    # Extract keywords from task prompt
    import re
    # Simple keyword extraction: find significant words
    words = re.findall(r'\b[a-zA-Z]{4,}\b', task_prompt.lower())
    # Remove common stop words
    stop_words = {'that', 'this', 'with', 'from', 'file', 'code', 'function', 'class', 'method', 'should', 'must', 'need', 'implement'}
    keywords = [w for w in words if w not in stop_words][:10]  # Top 10 keywords
    
    if not keywords:
        logger.warning("No meaningful keywords extracted from task prompt")
        # Return first chunks from first files as fallback
        retrieved_parts = []
        for file_path, code_content in list(context_files.items())[:top_k]:
            chunks = split_code(code_content)
            if chunks:
                retrieved_parts.append(f"From {file_path}:\n{chunks[0]}")
        return "\n\n".join(retrieved_parts)
    
    # Score chunks by keyword matches
    chunk_scores = []
    chunk_info = []
    
    for file_path, code_content in context_files.items():
        chunks = split_code(code_content)
        for idx, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            # Count keyword matches
            score = sum(1 for keyword in keywords if keyword in chunk_lower)
            chunk_scores.append(score)
            chunk_info.append((file_path, idx, chunk))
    
    if not chunk_scores:
        return ""
    
    # Get top-K chunks
    top_indices = np.argsort(chunk_scores)[-top_k:][::-1]
    
    retrieved_parts = []
    for idx in top_indices:
        file_path, chunk_idx, chunk_content = chunk_info[idx]
        score = chunk_scores[idx]
        retrieved_parts.append(
            f"From {file_path} (chunk {chunk_idx + 1}, keyword matches: {score}):\n{chunk_content}"
        )
    
    retrieved_text = "\n\n".join(retrieved_parts)
    logger.info(f"Retrieved {len(top_indices)} fragments using keyword matching")
    
    return retrieved_text


def retrieve_relevant(
    context_files: Dict[str, str],
    task_prompt: str,
    top_k: int = 5,
    method: str = 'embedding',
    model_name: str = 'all-MiniLM-L6-v2'
) -> str:
    """
    Main retrieval function that dispatches to appropriate method.
    
    Args:
        context_files: Dictionary mapping file paths to code content
        task_prompt: The task description/prompt
        top_k: Number of top fragments to retrieve
        method: Retrieval method ('embedding' or 'keyword')
        model_name: Name of the sentence transformer model (for embedding method)
        
    Returns:
        Formatted string with retrieved code fragments
    """
    if method == 'embedding':
        result = retrieve_relevant_embedding(context_files, task_prompt, top_k, model_name)
        # Fallback to keyword if embedding fails
        if not result and context_files:
            logger.warning("Embedding retrieval failed, falling back to keyword method")
            return retrieve_relevant_keyword(context_files, task_prompt, top_k)
        return result
    elif method == 'keyword':
        return retrieve_relevant_keyword(context_files, task_prompt, top_k)
    else:
        logger.warning(f"Unknown retrieval method: {method}. Falling back to keyword.")
        return retrieve_relevant_keyword(context_files, task_prompt, top_k)


def load_context_files_from_scenario(
    scenario: Dict,
    project_dir: Optional[Path] = None
) -> Dict[str, str]:
    """
    Load context file contents from scenario.
    
    This function handles different scenario formats:
    - If context_files contains file paths (strings), load from project directory
    - If context_files contains file contents (dict), return as-is
    
    Args:
        scenario: Scenario dictionary
        project_dir: Directory containing project files
        
    Returns:
        Dictionary mapping file paths to code content
    """
    context_files_list = scenario.get('context_files', [])
    
    if not context_files_list:
        logger.warning("No context files in scenario")
        return {}
    
    # Check if context_files is already a dict with contents
    if isinstance(context_files_list, dict):
        return context_files_list
    
    # If it's a list of file paths, try to load them
    if isinstance(context_files_list, list):
        context_files_dict = {}
        
        # Try to find project directory
        if project_dir is None:
            # Try to infer from scenario metadata or config
            scenario_id = scenario.get('id', '')
            # Project directory might be in metadata or we need to search
            # For now, return empty dict if we can't find files
            logger.warning(f"Cannot load context files without project directory. Scenario: {scenario_id}")
            return {}
        
        # Load files from project directory
        for file_path in context_files_list:
            file_full_path = project_dir / file_path
            if file_full_path.exists():
                try:
                    with open(file_full_path, 'r', encoding='utf-8') as f:
                        context_files_dict[file_path] = f.read()
                except Exception as e:
                    logger.warning(f"Failed to load file {file_path}: {e}")
            else:
                logger.warning(f"Context file not found: {file_full_path}")
        
        return context_files_dict
    
    return {}
