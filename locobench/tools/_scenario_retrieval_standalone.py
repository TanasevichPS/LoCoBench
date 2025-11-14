"""Standalone scenario retrieval functions - NO LangChain dependencies"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


def get_most_relevant_file_from_scenario(
    scenario_id: str,
    scenarios_dir: str = "data/output/scenarios",
    base_path: str = "/srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/data/generated"
) -> Optional[str]:
    """
    Get the most relevant file for a scenario using the scenario file.
    NO LangChain dependencies - pure Python implementation.
    """
    try:
        # Read scenario file
        scenario_path = Path(scenarios_dir) / f"{scenario_id}.json"
        scenario_path = scenario_path.resolve()
        
        if not scenario_path.exists():
            logger.error(f"Scenario file not found: {scenario_path}")
            logger.debug(f"Looking for scenario in directory: {scenarios_dir}")
            logger.debug(f"Scenario ID: {scenario_id}")
            return None
        
        logger.debug(f"Found scenario file: {scenario_path}")
        
        scenario_json = scenario_path.read_text(encoding='utf-8')
        scenario_data = json.loads(scenario_json)
        
        # Extract project name - must stop BEFORE task category
        # Format: {project_name}_{task_category}_{difficulty}_{number}
        parts = scenario_id.split('_')
        task_categories = [
            'architectural_understanding', 'cross_file_refactoring', 'feature_implementation',
            'bug_investigation', 'multi_session_development', 'code_comprehension',
            'integration_testing', 'security_analysis'
        ]
        difficulties = ['easy', 'medium', 'hard', 'expert']
        
        project_name = None
        
        # Simple strategy: split on task category
        # Format: {project_name}_{task_category}_{difficulty}_{number}
        # Example: java_web_social_medium_073_architectural_understanding_expert_01
        # Split on '_architectural_understanding' -> 'java_web_social_medium_073'
        
        for category in task_categories:
            if f'_{category}' in scenario_id:
                project_name = scenario_id.split(f'_{category}')[0]
                logger.debug(f"Found task category '{category}', project_name: {project_name}")
                break
        
        # Fallback: try finding difficulty
        if not project_name:
            for difficulty in difficulties:
                if f'_{difficulty}' in scenario_id:
                    # Find the last occurrence before the number
                    parts_before_difficulty = scenario_id.rsplit(f'_{difficulty}', 1)[0]
                    # Remove any trailing numbers/underscores
                    project_name = parts_before_difficulty.rstrip('_0123456789')
                    logger.debug(f"Found difficulty '{difficulty}', project_name: {project_name}")
                    break
        
        # Last fallback: remove last 3 parts
        if not project_name and len(parts) >= 4:
            project_name = '_'.join(parts[:-3])
            logger.debug(f"Fallback: Using first {len(parts)-3} parts as project_name: {project_name}")
        
        if not project_name:
            logger.error(f"Could not extract project name from scenario ID: {scenario_id}")
            return None
        
        logger.debug(f"Extracted project name: '{project_name}' from scenario ID: '{scenario_id}'")
        
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
        
        # Build full paths
        full_paths = []
        base_dir = Path(base_path) / project_name
        
        for rel_path in context_files:
            normalized = rel_path.replace('//', '/').replace('\\', '/').lstrip('/')
            full_path = base_dir / normalized
            full_paths.append(str(full_path))
        
        if not full_paths:
            logger.warning(f"No full paths generated for scenario {scenario_id}")
            return None
        
        # Find most relevant file using chunk-based analysis
        task_prompt = scenario_data.get('task_prompt', '')
        
        try:
            from .file_chunking import chunk_file_smart
            from .chunk_analysis import analyze_chunk_relevance
        except ImportError:
            # Fallback to simple keyword matching
            chunk_file_smart = None
            analyze_chunk_relevance = None
        
        best_file = None
        best_score = 0
        
        for file_path in full_paths[:10]:  # Limit to first 10 files
            path = Path(file_path)
            if not path.exists() or not path.is_file():
                continue
            
            try:
                content = path.read_text(encoding='utf-8', errors='ignore')
                
                # Use chunk-based analysis if available
                if chunk_file_smart and analyze_chunk_relevance:
                    chunks = chunk_file_smart(content, max_chunk_size=2000)
                    # Analyze top 3 chunks
                    chunk_scores = []
                    for chunk in chunks[:3]:  # Only check first 3 chunks
                        score = analyze_chunk_relevance(chunk['content'], task_prompt)
                        chunk_scores.append(score)
                    
                    # Use max chunk score as file score
                    if chunk_scores:
                        score = max(chunk_scores)
                    else:
                        score = 0.0
                else:
                    # Fallback to simple keyword matching
                    task_words = set(re.findall(r'\b\w+\b', task_prompt.lower()))
                    content_words = set(re.findall(r'\b\w+\b', content.lower()))
                    matches = len(task_words.intersection(content_words))
                    file_size_factor = min(1.0, 1000 / max(len(content), 1))
                    score = matches * file_size_factor
                
                if score > best_score:
                    best_score = score
                    best_file = file_path
            except Exception as e:
                logger.debug(f"Could not read file {file_path}: {e}")
                continue
        
        if best_file:
            logger.info(f"Found most relevant file for scenario {scenario_id}: {best_file} (score: {best_score:.3f})")
            return best_file
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
    NO LangChain dependencies - pure Python implementation.
    """
    try:
        scenario_path = Path(scenarios_dir) / f"{scenario_id}.json"
        scenario_path = scenario_path.resolve()
        
        if not scenario_path.exists():
            logger.error(f"Scenario file not found: {scenario_path}")
            return {}
        
        scenario_json = scenario_path.read_text(encoding='utf-8')
        scenario_data = json.loads(scenario_json)
        
        # Extract project name - must stop BEFORE task category
        # Format: {project_name}_{task_category}_{difficulty}_{number}
        # Example: java_web_social_medium_073_architectural_understanding_expert_01
        # Project name should be: java_web_social_medium_073
        
        parts = scenario_id.split('_')
        task_categories = [
            'architectural_understanding', 'cross_file_refactoring', 'feature_implementation',
            'bug_investigation', 'multi_session_development', 'code_comprehension',
            'integration_testing', 'security_analysis'
        ]
        difficulties = ['easy', 'medium', 'hard', 'expert']
        
        project_name = None
        
        # Simple strategy: split on task category
        # Format: {project_name}_{task_category}_{difficulty}_{number}
        for category in task_categories:
            if f'_{category}' in scenario_id:
                project_name = scenario_id.split(f'_{category}')[0]
                logger.debug(f"Found task category '{category}', project_name: {project_name}")
                break
        
        # Fallback: try finding difficulty
        if not project_name:
            for difficulty in difficulties:
                if f'_{difficulty}' in scenario_id:
                    parts_before_difficulty = scenario_id.rsplit(f'_{difficulty}', 1)[0]
                    project_name = parts_before_difficulty.rstrip('_0123456789')
                    logger.debug(f"Found difficulty '{difficulty}', project_name: {project_name}")
                    break
        
        # Last fallback: remove last 3 parts
        if not project_name and len(parts) >= 4:
            project_name = '_'.join(parts[:-3])
            logger.debug(f"Fallback: Using first {len(parts)-3} parts as project_name: {project_name}")
        
        if not project_name:
            logger.error(f"Could not extract project name from scenario ID: {scenario_id}")
            return {}
        
        logger.debug(f"Extracted project name: '{project_name}' from scenario ID: '{scenario_id}'")
        
        # Get context files
        context_files = scenario_data.get('context_files', [])
        logger.debug(f"Context files type: {type(context_files)}, count: {len(context_files) if isinstance(context_files, (list, dict)) else 'N/A'}")
        
        if isinstance(context_files, dict):
            # Already have content
            logger.debug(f"Context files is dict with {len(context_files)} items")
            return context_files
        elif isinstance(context_files, list):
            logger.debug(f"Context files is list with {len(context_files)} items")
            logger.debug(f"Project name: {project_name}, base_path: {base_path}")
            
            full_paths = []
            base_dir = Path(base_path) / project_name
            logger.debug(f"Base directory: {base_dir}")
            logger.debug(f"Base directory exists: {base_dir.exists()}")
            
            if not base_dir.exists():
                # Try alternative: maybe project_name needs to be extracted differently
                logger.warning(f"Base directory does not exist: {base_dir}")
                # List what's actually in base_path
                base_path_obj = Path(base_path)
                if base_path_obj.exists():
                    logger.debug(f"Contents of {base_path}: {list(base_path_obj.iterdir())[:5]}")
            
            for rel_path in context_files[:5]:  # Log first 5
                normalized = rel_path.replace('//', '/').replace('\\', '/').lstrip('/')
                full_path = base_dir / normalized
                full_paths.append(str(full_path))
                logger.debug(f"Built path: {rel_path} -> {full_path} (exists: {full_path.exists()})")
            
            # Build all paths
            for rel_path in context_files[5:]:
                normalized = rel_path.replace('//', '/').replace('\\', '/').lstrip('/')
                full_path = base_dir / normalized
                full_paths.append(str(full_path))
            
            logger.debug(f"Built {len(full_paths)} full paths from {len(context_files)} context files")
            
            # Read all files
            context_files_content = {}
            files_found = 0
            files_not_found = 0
            
            for full_path in full_paths:
                path = Path(full_path)
                if path.exists() and path.is_file():
                    try:
                        content = path.read_text(encoding='utf-8', errors='ignore')
                        # Try to get relative path
                        try:
                            rel_path = str(path.relative_to(Path(base_path) / project_name))
                        except ValueError:
                            # If relative path fails, try to find matching original path
                            rel_path = next((cf for cf in context_files if normalized in cf.replace('//', '/') or cf.replace('//', '/') in str(path)), str(path.name))
                        context_files_content[rel_path] = content
                        files_found += 1
                        if files_found <= 3:  # Log first 3
                            logger.debug(f"Successfully read file: {rel_path} ({len(content)} chars)")
                    except Exception as e:
                        logger.warning(f"Could not read file {full_path}: {e}")
                        files_not_found += 1
                else:
                    files_not_found += 1
                    if files_not_found <= 3:  # Log first 3 missing files
                        logger.debug(f"File does not exist: {full_path}")
            
            logger.info(f"Loaded {files_found} context files from {len(full_paths)} paths ({files_not_found} not found)")
            return context_files_content
        else:
            logger.warning(f"Context files is unexpected type: {type(context_files)}")
            return {}
            
    except Exception as e:
        logger.error(f"Error getting context files from scenario {scenario_id}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {}
