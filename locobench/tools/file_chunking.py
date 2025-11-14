"""File chunking utilities for MCP agent to avoid context overflow"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def chunk_file_content(content: str, chunk_size: int = 2000, overlap: int = 200) -> List[Dict[str, any]]:
    """
    Split file content into chunks with overlap.
    
    Args:
        content: File content to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of chunk dictionaries with 'content', 'start', 'end', 'chunk_index'
    """
    if len(content) <= chunk_size:
        return [{
            'content': content,
            'start': 0,
            'end': len(content),
            'chunk_index': 0
        }]
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(content):
        end = min(start + chunk_size, len(content))
        chunk_content = content[start:end]
        
        chunks.append({
            'content': chunk_content,
            'start': start,
            'end': end,
            'chunk_index': chunk_index,
            'size': len(chunk_content)
        })
        
        chunk_index += 1
        # Move start forward by chunk_size - overlap
        start += chunk_size - overlap
    
    return chunks


def chunk_file_by_lines(content: str, lines_per_chunk: int = 100, overlap_lines: int = 10) -> List[Dict[str, any]]:
    """
    Split file content into chunks by lines (better for code files).
    
    Args:
        content: File content to chunk
        lines_per_chunk: Number of lines per chunk
        overlap_lines: Number of lines to overlap between chunks
    
    Returns:
        List of chunk dictionaries
    """
    lines = content.split('\n')
    if len(lines) <= lines_per_chunk:
        return [{
            'content': content,
            'start_line': 0,
            'end_line': len(lines),
            'chunk_index': 0,
            'line_count': len(lines)
        }]
    
    chunks = []
    start_line = 0
    chunk_index = 0
    
    while start_line < len(lines):
        end_line = min(start_line + lines_per_chunk, len(lines))
        chunk_lines = lines[start_line:end_line]
        chunk_content = '\n'.join(chunk_lines)
        
        chunks.append({
            'content': chunk_content,
            'start_line': start_line + 1,  # 1-indexed for display
            'end_line': end_line,
            'chunk_index': chunk_index,
            'line_count': len(chunk_lines),
            'size': len(chunk_content)
        })
        
        chunk_index += 1
        # Move start forward by lines_per_chunk - overlap_lines
        start_line += lines_per_chunk - overlap_lines
    
    return chunks


def chunk_file_smart(content: str, max_chunk_size: int = 2000) -> List[Dict[str, any]]:
    """
    Smart chunking: try to chunk at natural boundaries (functions, classes, etc.).
    Falls back to line-based chunking if no natural boundaries found.
    
    Args:
        content: File content to chunk
        max_chunk_size: Maximum size of each chunk
    
    Returns:
        List of chunk dictionaries
    """
    # Try to find natural boundaries (function/class definitions)
    # Look for common patterns: def, class, function, etc.
    boundary_patterns = [
        r'^\s*(def|class|function|public|private|protected)\s+',  # Function/class definitions
        r'^\s*}\s*$',  # End of block
        r'^\s*#\s*---',  # Section separators
    ]
    
    lines = content.split('\n')
    if len(content) <= max_chunk_size:
        return [{
            'content': content,
            'start_line': 1,
            'end_line': len(lines),
            'chunk_index': 0,
            'line_count': len(lines)
        }]
    
    # Find potential split points
    split_points = [0]  # Start
    
    for i, line in enumerate(lines):
        for pattern in boundary_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                split_points.append(i)
                break
    
    split_points.append(len(lines))  # End
    
    # Remove duplicates and sort
    split_points = sorted(list(set(split_points)))
    
    # Create chunks
    chunks = []
    chunk_index = 0
    
    for i in range(len(split_points) - 1):
        start_line = split_points[i]
        end_line = split_points[i + 1]
        chunk_lines = lines[start_line:end_line]
        chunk_content = '\n'.join(chunk_lines)
        
        # If chunk is too large, split it further
        if len(chunk_content) > max_chunk_size:
            # Use line-based chunking for this section
            sub_chunks = chunk_file_by_lines(chunk_content, lines_per_chunk=50, overlap_lines=5)
            for sub_chunk in sub_chunks:
                sub_chunk['chunk_index'] = chunk_index
                sub_chunk['start_line'] = start_line + sub_chunk.get('start_line', 1) - 1
                chunks.append(sub_chunk)
                chunk_index += 1
        else:
            chunks.append({
                'content': chunk_content,
                'start_line': start_line + 1,
                'end_line': end_line,
                'chunk_index': chunk_index,
                'line_count': len(chunk_lines),
                'size': len(chunk_content)
            })
            chunk_index += 1
    
    return chunks


def get_file_chunks_summary(file_path: str, max_chunks: int = 10) -> str:
    """
    Get a summary of file chunks without loading full content.
    
    Args:
        file_path: Path to file
        max_chunks: Maximum number of chunks to summarize
    
    Returns:
        Summary string
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"
        
        # Read first part to estimate chunks
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            first_part = f.read(5000)
            total_size = path.stat().st_size
        
        # Estimate number of chunks
        estimated_chunks = max(1, total_size // 2000)
        
        return f"File: {path.name}\nSize: {total_size:,} bytes\nEstimated chunks: {estimated_chunks}\nFirst 500 chars: {first_part[:500]}..."
    except Exception as e:
        return f"Error reading file {file_path}: {e}"
