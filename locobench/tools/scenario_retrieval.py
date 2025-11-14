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
        # Import tool functions directly (avoid importing @tool decorators)
        # Read scenario file manually to avoid LangChain tool imports
        scenario_path = Path(scenarios_dir) / f"{scenario_id}.json"
        if not scenario_path.exists():
            logger.error(f"Scenario file not found: {scenario_path}")
            return None
        
        scenario_json = scenario_path.read_text(encoding='utf-8')
        scenario_data = json.loads(scenario_json)
        
        # Extract project name manually
        parts = scenario_id.split('_')
        task_categories = [
            'architectural_understanding', 'cross_file_refactoring', 'feature_implementation',
            'bug_investigation', 'multi_session_development', 'code_comprehension',
            'integration_testing', 'security_analysis'
        ]
        difficulties = ['easy', 'medium', 'hard', 'expert']
        
        project_name = None
        for i in range(len(parts) - 1, 0, -1):
            potential_category = '_'.join(parts[i:])
            if potential_category in task_categories:
                project_name = '_'.join(parts[:i])
                break
            if parts[i] in difficulties:
                project_name = '_'.join(parts[:i])
                break
        
        if not project_name and len(parts) >= 4:
            project_name = '_'.join(parts[:-3])
        
        if not project_name:
            logger.error(f"Could not extract project name from scenario ID: {scenario_id}")
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
        
        # Build full paths manually (avoid tool imports)
        full_paths = []
        base_dir = Path(base_path) / project_name
        
        for rel_path in context_files:
            # Normalize path separators (handle //)
            normalized = rel_path.replace('//', '/').replace('\\', '/')
            # Remove leading slash if present
            normalized = normalized.lstrip('/')
            full_path = base_dir / normalized
            full_paths.append(str(full_path))
        
        if not full_paths:
            logger.warning(f"No full paths generated for scenario {scenario_id}")
            return None
        
        # Find most relevant file manually (avoid tool imports)
        task_prompt = scenario_data.get('task_prompt', '')
        
        # Extract keywords from task_prompt
        import re
        task_words = set(re.findall(r'\b\w+\b', task_prompt.lower()))
        
        best_file = None
        best_score = 0
        
        # Limit number of files to check
        files_to_check = full_paths[:10]
        
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
                logger.debug(f"Could not read file {file_path}: {e}")
                continue
        
        most_relevant = best_file
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
        # Read scenario file manually (avoid LangChain tool imports)
        scenario_path = Path(scenarios_dir) / f"{scenario_id}.json"
        if not scenario_path.exists():
            logger.error(f"Scenario file not found: {scenario_path}")
            return {}
        
        scenario_json = scenario_path.read_text(encoding='utf-8')
        scenario_data = json.loads(scenario_json)
        
        # Extract project name manually
        parts = scenario_id.split('_')
        task_categories = [
            'architectural_understanding', 'cross_file_refactoring', 'feature_implementation',
            'bug_investigation', 'multi_session_development', 'code_comprehension',
            'integration_testing', 'security_analysis'
        ]
        difficulties = ['easy', 'medium', 'hard', 'expert']
        
        project_name = None
        for i in range(len(parts) - 1, 0, -1):
            potential_category = '_'.join(parts[i:])
            if potential_category in task_categories:
                project_name = '_'.join(parts[:i])
                break
            if parts[i] in difficulties:
                project_name = '_'.join(parts[:i])
                break
        
        if not project_name and len(parts) >= 4:
            project_name = '_'.join(parts[:-3])
        
        if not project_name:
            logger.error(f"Could not extract project name from scenario ID: {scenario_id}")
            return {}
        
        # Get context files
        context_files = scenario_data.get('context_files', [])
        if isinstance(context_files, dict):
            # Already have content
            return context_files
        elif isinstance(context_files, list):
            # Need to build paths and read files manually
            full_paths = []
            base_dir = Path(base_path) / project_name
            
            for rel_path in context_files:
                # Normalize path separators (handle //)
                normalized = rel_path.replace('//', '/').replace('\\', '/')
                # Remove leading slash if present
                normalized = normalized.lstrip('/')
                full_path = base_dir / normalized
                full_paths.append(str(full_path))
            
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
