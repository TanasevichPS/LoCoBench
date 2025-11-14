"""MCP Agent-based retrieval for evaluator using LangChain"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# Import chunking utilities
try:
    from .file_chunking import chunk_file_smart, chunk_file_by_lines, get_file_chunks_summary
except ImportError:
    # Fallback if import fails
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
    
    def chunk_file_by_lines(content, lines_per_chunk=100, overlap_lines=10):
        return chunk_file_smart(content, max_chunk_size=lines_per_chunk * 50)
    
    def get_file_chunks_summary(file_path, max_chunks=10):
        return f"File: {Path(file_path).name}"


def _select_file_by_keywords(task_prompt: str, file_paths: List[str]) -> Optional[str]:
    """Fallback: Select file using keyword matching."""
    if not file_paths:
        return None
    
    task_words = set(re.findall(r'\b\w+\b', task_prompt.lower()))
    best_file = None
    best_score = 0
    
    for file_path in file_paths[:10]:  # Limit to first 10 files
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            continue
        
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
            content_words = set(re.findall(r'\b\w+\b', content.lower()))
            matches = len(task_words.intersection(content_words))
            file_size_factor = min(1.0, 1000 / max(len(content), 1))
            score = matches * file_size_factor
            
            if score > best_score:
                best_score = score
                best_file = file_path
        except Exception:
            continue
    
    return best_file if best_file else (file_paths[0] if file_paths else None)


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
        
        # Read scenario file manually (avoid importing file_tools which has LangChain decorators)
        scenario_path = Path(scenarios_dir) / f"{scenario_id}.json"
        scenario_path = scenario_path.resolve()
        
        if not scenario_path.exists():
            logger.error(f"Scenario file not found: {scenario_path}")
            return None
        
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
        
        # Build full paths manually (avoid importing file_tools)
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
        
        # Create MCP agent
        model = ChatOpenAI(
            model=mcp_model,
            temperature=0.0,
            base_url=mcp_base_url,
            api_key=mcp_api_key,
            streaming=False,
            timeout=30.0
        )
        
        # Create a tool that reads file chunks (to avoid context overflow)
        @tool
        def read_file_chunk(file_index: int, chunk_index: int = 0) -> str:
            """Read a specific chunk from a file by index.
            
            Args:
                file_index: Index of the file to read (0-based)
                chunk_index: Index of the chunk to read (0-based, default: 0 for first chunk)
            
            Returns:
                Chunk content with metadata or error message
            """
            if 0 <= file_index < len(full_paths):
                file_path = full_paths[file_index]
                try:
                    path = Path(file_path)
                    if not path.exists():
                        return f"Error: File '{file_path}' does not exist"
                    
                    content = path.read_text(encoding='utf-8', errors='ignore')
                    # Chunk the file
                    chunks = chunk_file_smart(content, max_chunk_size=2000)
                    
                    if chunk_index < len(chunks):
                        chunk = chunks[chunk_index]
                        return f"File: {path.name}\nChunk {chunk_index + 1}/{len(chunks)}\nLines {chunk.get('start_line', '?')}-{chunk.get('end_line', '?')}\n\n{chunk['content']}"
                    else:
                        return f"Error: Chunk index {chunk_index} out of range. File has {len(chunks)} chunks."
                except Exception as e:
                    return f"Error reading file '{file_path}': {str(e)}"
            else:
                return f"Error: Invalid file index {file_index}. Available files: {len(full_paths)}"
        
        @tool
        def get_file_chunk_count(file_index: int) -> str:
            """Get the number of chunks in a file.
            
            Args:
                file_index: Index of the file (0-based)
            
            Returns:
                Number of chunks or error message
            """
            if 0 <= file_index < len(full_paths):
                file_path = full_paths[file_index]
                try:
                    path = Path(file_path)
                    if not path.exists():
                        return f"Error: File '{file_path}' does not exist"
                    
                    content = path.read_text(encoding='utf-8', errors='ignore')
                    chunks = chunk_file_smart(content, max_chunk_size=2000)
                    return f"File {file_index} ({path.name}) has {len(chunks)} chunks"
                except Exception as e:
                    return f"Error: {str(e)}"
            else:
                return f"Error: Invalid file index {file_index}"
        
        @tool
        def list_available_files() -> str:
            """List all available files with their indices for this scenario.
            
            Returns:
                Formatted string listing all files with indices
            """
            file_list = []
            for i, file_path in enumerate(full_paths):
                file_name = Path(file_path).name
                file_list.append(f"{i}: {file_name} ({file_path})")
            return "\n".join(file_list)
        
        # Create agent with chunked file reading tools
        # Limit task_prompt length to avoid context overflow
        task_prompt_short = task_prompt[:300] if len(task_prompt) > 300 else task_prompt
        
        agent = create_agent(
            model,
            tools=[read_file_chunk, get_file_chunk_count, list_available_files],
            system_prompt=f"""You are a helpful assistant that finds the most relevant file for a task.

Task prompt: {task_prompt_short}

Your goal is to identify which file (by index) is most relevant to the task prompt. 
Files are split into chunks to avoid context overflow.

Process:
1. Use list_available_files() to see all available files
2. For 2-3 most promising files:
   - Use get_file_chunk_count(file_index) to see how many chunks each file has
   - Use read_file_chunk(file_index, chunk_index=0) to read the first chunk of each file
   - If needed, read chunk_index=1, 2, etc. for more context (but limit to 2-3 chunks per file)
3. Compare the chunks and determine which file is most relevant
4. Return ONLY the file index number (0-based) of the most relevant file.

Important: Read only 2-3 files and 1-2 chunks per file to avoid context overflow."""
        )
        
        # Invoke agent to find relevant file
        try:
            result = agent.invoke({
                "messages": [{
                    "role": "user",
                    "content": f"Find the most relevant file for this task: {task_prompt_short}. Use list_available_files(), then get_file_chunk_count() and read_file_chunk() for 2-3 promising files. Read only 1-2 chunks per file. Return only the file index number."
                }]
            })
            
            # Extract file index from agent response
            # The agent should return something like "File 2 is most relevant" or just "2"
            if isinstance(result, dict):
                messages = result.get('messages', [])
                if messages:
                    response_content = str(messages[-1].content)
                else:
                    response_content = str(result)
            else:
                response_content = str(result)
            
            # Try to extract index from response
            import re
            # Look for numbers in the response
            numbers = re.findall(r'\b(\d+)\b', response_content)
            if numbers:
                # Try each number to see if it's a valid index
                for num_str in numbers:
                    file_index = int(num_str)
                    if 0 <= file_index < len(full_paths):
                        logger.info(f"MCP agent selected file index {file_index}: {full_paths[file_index]}")
                        return full_paths[file_index]
            
            # Fallback: return first file
            logger.warning(f"Could not parse file index from agent response: {response_content}")
            logger.info(f"Using first file as fallback: {full_paths[0]}")
            return full_paths[0] if full_paths else None
        except Exception as e:
            logger.error(f"Error invoking MCP agent: {e}")
            # Fallback: use keyword-based matching instead
            logger.info("Falling back to keyword-based file selection")
            return _select_file_by_keywords(task_prompt, full_paths)
        
    except Exception as e:
        logger.error(f"Error using MCP agent for retrieval: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
