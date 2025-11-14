"""MCP Agent-based retrieval for evaluator using LangChain"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_most_relevant_file_with_mcp_agent(
    scenario_id: str,
    task_prompt: str,
    scenarios_dir: str = "data/output/scenarios",
    base_path: str = "/srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/data/generated",
    mcp_base_url: str = "http://10.199.178.176:8080/v1",
    mcp_api_key: str = "111",
    mcp_model: str = "gpt-oss"
) -> Optional[str]:
    """
    Get the most relevant file using LangChain MCP agent.
    
    Args:
        scenario_id: The scenario ID
        task_prompt: The task prompt
        scenarios_dir: Directory containing scenario files
        base_path: Base path for generated projects
        mcp_base_url: Base URL for MCP model
        mcp_api_key: API key for MCP model
        mcp_model: Model name for MCP agent
    
    Returns:
        Path to the most relevant file, or None if not found
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from langchain.agents import create_agent
        from .file_tools import (
            read_scenario_file,
            extract_project_name_from_scenario_id,
            build_full_file_paths,
            read_file
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
        
        # Build full paths
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
        
        # Create MCP agent
        model = ChatOpenAI(
            model=mcp_model,
            temperature=0.0,
            base_url=mcp_base_url,
            api_key=mcp_api_key,
            streaming=False,
            timeout=30.0
        )
        
        # Create a tool that reads files from the full_paths list
        @tool
        def read_relevant_file(file_index: int) -> str:
            """Read a file by index from the available files for this scenario.
            
            Args:
                file_index: Index of the file to read (0-based)
            
            Returns:
                File contents or error message
            """
            if 0 <= file_index < len(full_paths):
                file_path = full_paths[file_index]
                return read_file(file_path)
            else:
                return f"Error: Invalid file index {file_index}. Available files: {len(full_paths)}"
        
        # Create agent with the file reading tool
        agent = create_agent(
            model,
            tools=[read_relevant_file],
            system_prompt=f"""You are a helpful assistant that finds the most relevant file for a task.

Available files for scenario {scenario_id}:
{chr(10).join([f"{i}: {Path(p).name}" for i, p in enumerate(full_paths)])}

Task prompt: {task_prompt}

Your goal is to identify which file (by index) is most relevant to the task prompt. 
Read the files and determine which one contains the most relevant information for completing the task.
Be concise and accurate."""
        )
        
        # Invoke agent to find relevant file
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Find the most relevant file for this task: {task_prompt}. Return the file index."
            }]
        })
        
        # Extract file index from agent response
        # The agent should return something like "File 2 is most relevant" or just "2"
        response_content = str(result.get('messages', [])[-1].content if isinstance(result, dict) else result)
        
        # Try to extract index from response
        import re
        index_match = re.search(r'\b(\d+)\b', response_content)
        if index_match:
            file_index = int(index_match.group(1))
            if 0 <= file_index < len(full_paths):
                logger.info(f"MCP agent selected file index {file_index}: {full_paths[file_index]}")
                return full_paths[file_index]
        
        # Fallback: return first file
        logger.warning(f"Could not parse file index from agent response: {response_content}")
        logger.info(f"Using first file as fallback: {full_paths[0]}")
        return full_paths[0] if full_paths else None
        
    except Exception as e:
        logger.error(f"Error using MCP agent for retrieval: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
