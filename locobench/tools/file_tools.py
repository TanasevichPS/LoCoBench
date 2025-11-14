"""Simple file tools for MCP integration with LangChain"""

from pathlib import Path
from typing import Optional
from langchain_core.tools import tool


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file.
    
    Args:
        file_path: The path to the file to read (relative to workspace root)
    
    Returns:
        The contents of the file as a string
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist"
        return path.read_text(encoding='utf-8')
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file.
    
    Args:
        file_path: The path to the file to write (relative to workspace root)
        content: The content to write to the file
    
    Returns:
        Success message or error message
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return f"Successfully wrote to '{file_path}'"
    except Exception as e:
        return f"Error writing file '{file_path}': {str(e)}"


@tool
def list_directory(directory_path: str) -> str:
    """List files and directories in a given directory.
    
    Args:
        directory_path: The path to the directory to list (relative to workspace root)
    
    Returns:
        A formatted string listing files and directories
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            return f"Error: Directory '{directory_path}' does not exist"
        if not path.is_dir():
            return f"Error: '{directory_path}' is not a directory"
        
        items = []
        for item in sorted(path.iterdir()):
            if item.is_dir():
                items.append(f"[DIR]  {item.name}/")
            else:
                items.append(f"[FILE] {item.name}")
        
        if not items:
            return f"Directory '{directory_path}' is empty"
        
        return "\n".join(items)
    except Exception as e:
        return f"Error listing directory '{directory_path}': {str(e)}"


@tool
def file_exists(file_path: str) -> str:
    """Check if a file exists.
    
    Args:
        file_path: The path to the file to check (relative to workspace root)
    
    Returns:
        "File exists" or "File does not exist" message
    """
    try:
        path = Path(file_path)
        if path.exists():
            return f"File '{file_path}' exists"
        else:
            return f"File '{file_path}' does not exist"
    except Exception as e:
        return f"Error checking file '{file_path}': {str(e)}"


# List of all file tools
file_tools = [
    read_file,
    write_file,
    list_directory,
    file_exists,
]
