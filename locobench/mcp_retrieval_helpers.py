"""
Helper functions for MCP retrieval to load files from project directory
"""

from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def load_files_from_project_dir(project_dir: Path) -> Dict[str, str]:
    """
    Load files from project directory for MCP tools.
    
    Args:
        project_dir: Path to project directory
    
    Returns:
        Dictionary mapping file paths to file contents
    """
    if not project_dir or not project_dir.exists():
        return {}
    
    try:
        from ..retrieval import _collect_project_code_files
        
        project_files = _collect_project_code_files(project_dir)
        files_dict = {
            file_info["path"]: file_info["content"]
            for file_info in project_files
        }
        logger.debug(f"Loaded {len(files_dict)} files from {project_dir}")
        return files_dict
    except Exception as e:
        logger.warning(f"Failed to load files from project_dir {project_dir}: {e}")
        return {}


def get_files_for_tool(context_files: Dict[str, str], project_dir: Path) -> Dict[str, str]:
    """
    Get files for tool execution, loading from project_dir if context_files is empty.
    
    Args:
        context_files: Dictionary of context files (may be empty)
        project_dir: Path to project directory
    
    Returns:
        Dictionary of files to search
    """
    if context_files:
        return context_files
    
    if project_dir and project_dir.exists():
        return load_files_from_project_dir(project_dir)
    
    return {}
