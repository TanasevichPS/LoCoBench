"""Simple file tools for MCP integration with LangChain"""

import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Any
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


@tool
def read_scenario_file(scenario_id: str, scenarios_dir: str = "data/output/scenarios") -> str:
    """Read a scenario JSON file based on scenario ID.
    
    Args:
        scenario_id: The scenario ID (e.g., 'c_api_gateway_easy_009_architectural_understanding_expert_01')
        scenarios_dir: Directory containing scenario files (default: 'data/output/scenarios')
    
    Returns:
        JSON string with scenario data or error message
    """
    try:
        scenario_path = Path(scenarios_dir) / f"{scenario_id}.json"
        if not scenario_path.exists():
            return f"Error: Scenario file '{scenario_path}' does not exist"
        return scenario_path.read_text(encoding='utf-8')
    except Exception as e:
        return f"Error reading scenario file '{scenario_id}': {str(e)}"


@tool
def extract_project_name_from_scenario_id(scenario_id: str) -> str:
    """Extract the unique project name from a scenario ID.
    
    Args:
        scenario_id: The scenario ID (e.g., 'c_api_gateway_easy_009_architectural_understanding_expert_01')
    
    Returns:
        Project name (e.g., 'c_api_gateway_easy_009') or error message
    """
    try:
        # Pattern: {project_name}_{task_category}_{difficulty}_{number}
        # Extract everything before the last two underscores (task_category and difficulty)
        parts = scenario_id.split('_')
        if len(parts) < 3:
            return f"Error: Invalid scenario ID format '{scenario_id}'"
        
        # Find task categories and difficulties to identify where project name ends
        task_categories = [
            'architectural_understanding', 'cross_file_refactoring', 'feature_implementation',
            'bug_investigation', 'multi_session_development', 'code_comprehension',
            'integration_testing', 'security_analysis'
        ]
        difficulties = ['easy', 'medium', 'hard', 'expert']
        
        # Try to find where task_category starts
        for i in range(len(parts) - 1, 0, -1):
            # Check if this part or combination with next is a task_category
            potential_category = '_'.join(parts[i:])
            if potential_category in task_categories:
                return '_'.join(parts[:i])
            
            # Check if this part is a difficulty
            if parts[i] in difficulties:
                # Project name is everything before this difficulty
                return '_'.join(parts[:i])
        
        # Fallback: return everything except last 3 parts (assuming format: name_category_difficulty_number)
        if len(parts) >= 4:
            return '_'.join(parts[:-3])
        
        return f"Error: Could not extract project name from '{scenario_id}'"
    except Exception as e:
        return f"Error extracting project name: {str(e)}"


@tool
def build_full_file_paths(
    project_name: str,
    context_files: str,  # JSON string or comma-separated list
    base_path: str = "/srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/data/generated"
) -> str:
    """Build full file paths from context_files list by combining with base path.
    
    Args:
        project_name: Unique project name (e.g., 'c_api_gateway_easy_009')
        context_files: List of relative file paths from scenario (e.g., ['EduGate_ScholarLink//src//main.c'])
        base_path: Base path for generated projects (default: '/srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/data/generated')
    
    Returns:
        JSON string with list of full file paths or error message
    """
    try:
        # Parse context_files if it's a JSON string
        if isinstance(context_files, str):
            try:
                parsed = json.loads(context_files)
                if isinstance(parsed, list):
                    context_files_list = parsed
                else:
                    # Try comma-separated
                    context_files_list = [f.strip() for f in context_files.split(',')]
            except:
                # Assume comma-separated string
                context_files_list = [f.strip() for f in context_files.split(',')]
        else:
            context_files_list = context_files
        
        full_paths = []
        base_dir = Path(base_path) / project_name
        
        for rel_path in context_files_list:
            # Normalize path separators (handle //)
            normalized = rel_path.replace('//', '/').replace('\\', '/')
            # Remove leading slash if present
            normalized = normalized.lstrip('/')
            full_path = base_dir / normalized
            full_paths.append(str(full_path))
        
        return json.dumps({"full_paths": full_paths})
    except Exception as e:
        return f"Error building file paths: {str(e)}"


@tool
def find_most_relevant_file(
    task_prompt: str,
    file_paths: str,  # JSON string or comma-separated list
    max_files_to_read: int = 10
) -> str:
    """Find the most relevant file for a task prompt by reading files and matching keywords.
    
    Args:
        task_prompt: The task prompt from the scenario
        file_paths: List of full file paths to search
        max_files_to_read: Maximum number of files to read for relevance check (default: 10)
    
    Returns:
        JSON string with the most relevant file path and score, or error message
    """
    try:
        # Parse file_paths if it's a JSON string
        if isinstance(file_paths, str):
            try:
                parsed = json.loads(file_paths)
                if isinstance(parsed, list):
                    file_paths_list = parsed
                elif isinstance(parsed, dict) and 'full_paths' in parsed:
                    file_paths_list = parsed['full_paths']
                else:
                    # Try comma-separated
                    file_paths_list = [f.strip() for f in file_paths.split(',')]
            except:
                # Assume comma-separated string
                file_paths_list = [f.strip() for f in file_paths.split(',')]
        else:
            file_paths_list = file_paths
        
        if not file_paths_list:
            return json.dumps({"error": "No file paths provided"})
        
        # Extract keywords from task_prompt
        task_words = set(re.findall(r'\b\w+\b', task_prompt.lower()))
        
        best_file = None
        best_score = 0
        
        # Limit number of files to check
        files_to_check = file_paths_list[:max_files_to_read]
        
        for file_path in files_to_check:
            path = Path(file_path)
            if not path.exists() or not path.is_file():
                continue
            
            try:
                content = path.read_text(encoding='utf-8', errors='ignore')
                content_words = set(re.findall(r'\b\w+\b', content.lower()))
                
                # Simple relevance score: count matching words
                matches = len(task_words.intersection(content_words))
                # Normalize by file size (prefer smaller files with more matches)
                file_size_factor = min(1.0, 1000 / max(len(content), 1))
                score = matches * file_size_factor
                
                if score > best_score:
                    best_score = score
                    best_file = file_path
            except Exception as e:
                # Skip files that can't be read
                continue
        
        if best_file:
            return json.dumps({
                "most_relevant_file": best_file,
                "relevance_score": best_score
            })
        else:
            # Fallback: return first existing file
            for file_path in file_paths_list:
                if Path(file_path).exists():
                    return json.dumps({
                        "most_relevant_file": file_path,
                        "relevance_score": 0.0,
                        "note": "No relevance match found, returning first existing file"
                    })
            
            return json.dumps({"error": "No accessible files found"})
    except Exception as e:
        return f"Error finding relevant file: {str(e)}"


@tool
def get_scenario_context_for_evaluation(scenario_id: str) -> str:
    """Get context files for evaluation from a scenario file.
    This is a convenience function that combines reading scenario, extracting project name,
    building paths, and finding the most relevant file.
    
    Args:
        scenario_id: The scenario ID (e.g., 'c_api_gateway_easy_009_architectural_understanding_expert_01')
    
    Returns:
        JSON string with scenario data, project name, full paths, and most relevant file
    """
    try:
        # Read scenario file
        scenario_json = read_scenario_file(scenario_id)
        if scenario_json.startswith("Error"):
            return scenario_json
        
        scenario_data = json.loads(scenario_json)
        
        # Extract project name
        project_name = extract_project_name_from_scenario_id(scenario_id)
        if project_name.startswith("Error"):
            return project_name
        
        # Get context files
        context_files = scenario_data.get('context_files', [])
        if isinstance(context_files, dict):
            context_files = list(context_files.keys())
        elif not isinstance(context_files, list):
            context_files = []
        
        # Build full paths
        full_paths_json = build_full_file_paths(project_name, context_files)
        if full_paths_json.startswith("Error"):
            return full_paths_json
        
        full_paths_data = json.loads(full_paths_json)
        full_paths = full_paths_data.get('full_paths', [])
        
        # Find most relevant file
        task_prompt = scenario_data.get('task_prompt', '')
        relevant_file_json = find_most_relevant_file(task_prompt, full_paths)
        relevant_file_data = json.loads(relevant_file_json)
        
        return json.dumps({
            "scenario_id": scenario_id,
            "project_name": project_name,
            "task_category": scenario_data.get('task_category'),
            "difficulty": scenario_data.get('difficulty'),
            "task_prompt": task_prompt,
            "context_files": context_files,
            "full_paths": full_paths,
            "most_relevant_file": relevant_file_data.get('most_relevant_file'),
            "relevance_score": relevant_file_data.get('relevance_score', 0.0)
        }, indent=2)
    except Exception as e:
        return f"Error getting scenario context: {str(e)}"


# List of all file tools
file_tools = [
    read_file,
    write_file,
    list_directory,
    file_exists,
    read_scenario_file,
    extract_project_name_from_scenario_id,
    build_full_file_paths,
    find_most_relevant_file,
    get_scenario_context_for_evaluation,
]
