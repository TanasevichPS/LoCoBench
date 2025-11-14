"""Chunk analysis utilities for finding most relevant chunks"""

import re
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def analyze_chunk_relevance(chunk_content: str, task_prompt: str) -> float:
    """
    Analyze how relevant a chunk is to the task prompt.
    
    Args:
        chunk_content: Content of the chunk
        task_prompt: The task prompt
    
    Returns:
        Relevance score (0.0 to 1.0)
    """
    # Extract keywords from task prompt
    task_words = set(re.findall(r'\b\w+\b', task_prompt.lower()))
    
    # Extract keywords from chunk
    chunk_words = set(re.findall(r'\b\w+\b', chunk_content.lower()))
    
    # Calculate overlap
    matches = len(task_words.intersection(chunk_words))
    
    # Normalize by task prompt size
    if len(task_words) == 0:
        return 0.0
    
    base_score = matches / len(task_words)
    
    # Boost score if chunk contains important keywords (function names, class names, etc.)
    important_patterns = [
        r'\b(def|class|function|method|interface|abstract)\s+(\w+)',  # Definitions
        r'\b(import|from|require|include)\s+',  # Imports
        r'\b(return|yield|throw|raise)\s+',  # Control flow
    ]
    
    pattern_matches = 0
    for pattern in important_patterns:
        if re.search(pattern, chunk_content, re.IGNORECASE):
            pattern_matches += 1
    
    # Boost score based on pattern matches
    pattern_boost = min(0.3, pattern_matches * 0.1)
    
    # Normalize by chunk size (prefer smaller chunks with more matches)
    size_factor = min(1.0, 2000 / max(len(chunk_content), 1))
    
    final_score = min(1.0, base_score + pattern_boost) * size_factor
    
    return final_score


def find_most_relevant_chunks(
    file_chunks: List[Dict],
    task_prompt: str,
    max_chunks: int = 3
) -> List[Tuple[int, float]]:
    """
    Find the most relevant chunks from a file.
    
    Args:
        file_chunks: List of chunk dictionaries
        task_prompt: The task prompt
        max_chunks: Maximum number of chunks to return
    
    Returns:
        List of tuples (chunk_index, relevance_score) sorted by relevance
    """
    chunk_scores = []
    
    for chunk in file_chunks:
        chunk_content = chunk.get('content', '')
        score = analyze_chunk_relevance(chunk_content, task_prompt)
        chunk_scores.append((chunk['chunk_index'], score))
    
    # Sort by score (descending)
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top chunks
    return chunk_scores[:max_chunks]


def get_best_chunks_from_files(
    files_content: Dict[str, str],
    task_prompt: str,
    max_chunks_per_file: int = 2,
    max_total_chunks: int = 5
) -> List[Dict]:
    """
    Get the best chunks from multiple files.
    
    Args:
        files_content: Dictionary mapping file paths to file contents
        task_prompt: The task prompt
        max_chunks_per_file: Maximum chunks to take from each file
        max_total_chunks: Maximum total chunks to return
    
    Returns:
        List of chunk dictionaries with file path and relevance score
    """
    try:
        from .file_chunking import chunk_file_smart
    except ImportError:
        # Fallback chunking
        def chunk_file_smart(content, max_chunk_size=2000):
            lines = content.split('\n')
            chunks = []
            lines_per_chunk = 100
            for i in range(0, len(lines), lines_per_chunk):
                chunk_lines = lines[i:i+lines_per_chunk]
                chunks.append({
                    'content': '\n'.join(chunk_lines),
                    'start_line': i + 1,
                    'end_line': min(i + lines_per_chunk, len(lines)),
                    'chunk_index': len(chunks),
                    'line_count': len(chunk_lines)
                })
            return chunks
    
    all_chunks = []
    
    for file_path, content in files_content.items():
        chunks = chunk_file_smart(content, max_chunk_size=2000)
        
        # Find most relevant chunks from this file
        relevant_chunks = find_most_relevant_chunks(chunks, task_prompt, max_chunks_per_file)
        
        for chunk_idx, score in relevant_chunks:
            chunk = chunks[chunk_idx]
            all_chunks.append({
                'file_path': file_path,
                'chunk_index': chunk_idx,
                'relevance_score': score,
                'content': chunk['content'],
                'start_line': chunk.get('start_line', 0),
                'end_line': chunk.get('end_line', 0),
            })
    
    # Sort all chunks by relevance score
    all_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Return top chunks
    return all_chunks[:max_total_chunks]
