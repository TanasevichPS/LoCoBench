"""Helper functions for scenario-based file retrieval in evaluator"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def get_most_relevant_file_from_scenario(
    scenario_id: str,
    scenarios_dir: str = "data/output/scenarios",
    base_path: str = "/srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/data/generated"
) -> Optional[str]:
    """
    Get the most relevant file for a scenario using the scenario file.
    
    Args:
        scenario_id: The scenario ID (e.g., 'c_api_gateway_easy_009_architectural_understanding_expert_01')
        scenarios_dir: Directory containing scenario files
        base_path: Base path for generated projects
    
    Returns:
        Path to the most relevant file, or None if not found
    """
    try:
        # Import tools here to avoid circular imports
        from .file_tools import (
            read_scenario_file,
            extract_project_name_from_scenario_id,
            build_full_file_paths,
            find_most_relevant_file
        )
        
        # Read scenario file
        scenario_json = read_scenario_file(scenario_id, scenarios_dir)
        if scenario_json.startswith("Error"):
            logger.error(f"Failed to read scenario file: {scenario_json}")
            return None
        
        scenario_data = json.loads(scenario_json)
        
        # Extract project name
        project_name = extract_project_name_from_scenario_id(scenario_id)
        if project_name.startswith("Error"):
            logger.error(f"Failed to extract project name: {project_name}")
            return None
        
        # Get context files
        context_files = scenario_data.get('context_files', [])
        if isinstance(context_files, dict):
            context_files = list(context_files.keys())
        elif not isinstance(context_files, list):
            logger.warning(f"No context_files found in scenario {scenario_id}")
            return None
        
        if not context_files:
            logger.warning(f"Empty context_files list in scenario {scenario_id}")
            return None
        
        # Build full paths (pass as JSON string)
        context_files_json = json.dumps(context_files)
        full_paths_json = build_full_file_paths(project_name, context_files_json, base_path)
        if full_paths_json.startswith("Error"):
            logger.error(f"Failed to build file paths: {full_paths_json}")
            return None
        
        full_paths_data = json.loads(full_paths_json)
        full_paths = full_paths_data.get('full_paths', [])
        
        if not full_paths:
            logger.warning(f"No full paths generated for scenario {scenario_id}")
            return None
        
        # Find most relevant file (pass as JSON string)
        task_prompt = scenario_data.get('task_prompt', '')
        full_paths_json = json.dumps(full_paths)
        relevant_file_json = find_most_relevant_file(task_prompt, full_paths_json)
        relevant_file_data = json.loads(relevant_file_json)
        
        most_relevant = relevant_file_data.get('most_relevant_file')
        if most_relevant:
            logger.info(f"Found most relevant file for scenario {scenario_id}: {most_relevant}")
            return most_relevant
        else:
            logger.warning(f"No relevant file found for scenario {scenario_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting relevant file from scenario {scenario_id}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def get_context_files_from_scenario(
    scenario_id: str,
    scenarios_dir: str = "data/output/scenarios",
    base_path: str = "/srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/data/generated"
) -> Dict[str, str]:
    """
    Get all context files content from a scenario.
    
    Args:
        scenario_id: The scenario ID
        scenarios_dir: Directory containing scenario files
        base_path: Base path for generated projects
    
    Returns:
        Dictionary mapping file paths to file contents
    """
    try:
        from .file_tools import (
            read_scenario_file,
            extract_project_name_from_scenario_id,
            build_full_file_paths
        )
        
        # Read scenario file
        scenario_json = read_scenario_file(scenario_id, scenarios_dir)
        if scenario_json.startswith("Error"):
            logger.error(f"Failed to read scenario file: {scenario_json}")
            return {}
        
        scenario_data = json.loads(scenario_json)
        
        # Extract project name
        project_name = extract_project_name_from_scenario_id(scenario_id)
        if project_name.startswith("Error"):
            logger.error(f"Failed to extract project name: {project_name}")
            return {}
        
        # Get context files
        context_files = scenario_data.get('context_files', [])
        if isinstance(context_files, dict):
            # Already have content
            return context_files
        elif isinstance(context_files, list):
            # Need to build paths and read files
            context_files_json = json.dumps(context_files)
            full_paths_json = build_full_file_paths(project_name, context_files_json, base_path)
            if full_paths_json.startswith("Error"):
                logger.error(f"Failed to build file paths: {full_paths_json}")
                return {}
            
            full_paths_data = json.loads(full_paths_json)
            full_paths = full_paths_data.get('full_paths', [])
            
            # Read all files
            context_files_content = {}
            for full_path in full_paths:
                path = Path(full_path)
                if path.exists() and path.is_file():
                    try:
                        content = path.read_text(encoding='utf-8', errors='ignore')
                        # Use relative path as key
                        rel_path = str(path.relative_to(Path(base_path) / project_name))
                        context_files_content[rel_path] = content
                    except Exception as e:
                        logger.warning(f"Could not read file {full_path}: {e}")
            
            return context_files_content
        else:
            return {}
            
    except Exception as e:
        logger.error(f"Error getting context files from scenario {scenario_id}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {}
