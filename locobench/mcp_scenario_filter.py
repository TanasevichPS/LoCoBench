"""
MCP Tool for LoCoBench - Selects Relevant Information from Code Files

This module provides an MCP tool that uses LLM to intelligently select
relevant code snippets and information from context files for scenarios.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from .core.config import Config

logger = logging.getLogger(__name__)

# LangChain imports - optional, only needed for LLM-based content selection
_LANGCHAIN_AVAILABLE = False
_ChatOpenAI = None

def _try_import_langchain():
    """Try to import LangChain components for LLM-based content selection"""
    global _LANGCHAIN_AVAILABLE, _ChatOpenAI
    
    if _LANGCHAIN_AVAILABLE:
        return True
    
    try:
        from langchain_openai import ChatOpenAI as _ChatOpenAI_impl
        _ChatOpenAI = _ChatOpenAI_impl
        _LANGCHAIN_AVAILABLE = True
        return True
    except (ImportError, ModuleNotFoundError) as e:
        logger.debug(f"LangChain not available: {e}")
        return False


class LoCoBenchMCPTool:
    """MCP tool for selecting relevant information from code files"""
    
    def __init__(self, config: Config, base_url: str = None, api_key: str = None, model: str = None):
        """
        Initialize the MCP tool
        
        Args:
            config: LoCoBench configuration object
            base_url: Base URL for the OpenAI-compatible API (defaults to config.mcp_filter.base_url)
            api_key: API key for authentication (defaults to config.mcp_filter.api_key)
            model: Model name for content selection (defaults to config.mcp_filter.model)
        """
        self.config = config
        
        # Use config values as defaults
        self.base_url = base_url or config.mcp_filter.base_url
        self.api_key = api_key or config.mcp_filter.api_key
        model_name = model or config.mcp_filter.model
        
        # Initialize LLM model if LangChain is available and enabled
        self.model = None
        if config.mcp_filter.use_llm_selection and _try_import_langchain():
            try:
                self.model = _ChatOpenAI(
                    model=model_name,
                    temperature=0.0,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    streaming=False,
                    timeout=120.0  # Increased timeout for large files
                )
                logger.info("MCP tool initialized with LLM for content selection")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM model: {e}. Will use full file content.")
                self.model = None
        else:
            logger.debug("MCP tool initialized without LLM - will return full file content")
        
        # Configuration for content selection
        self.max_file_size_for_llm = 100000  # 100KB - files larger than this use full content
        self.max_prompt_size = 30000  # Limit prompt size to avoid timeouts
    
    def _extract_project_dir_from_id(self, scenario_id: str) -> str:
        """Extract project directory name from scenario ID.
        
        Example: "c_api_gateway_easy_009_bug_investigation_expert_01" -> "c_api_gateway_easy_009"
        
        Args:
            scenario_id: Scenario ID string
            
        Returns:
            Project directory name
        """
        if not scenario_id:
            return ""
        
        # Split by underscore
        parts = scenario_id.split('_')
        
        # The project directory is typically the first few parts before task_category
        # Common pattern: {lang}_{project_name}_{complexity}_{number}_{task_category}_{difficulty}_{instance}
        # We want: {lang}_{project_name}_{complexity}_{number}
        
        # Find where task_category starts (common task categories)
        task_categories = [
            'architectural_understanding', 'cross_file_refactoring', 'feature_implementation',
            'bug_investigation', 'multi_session_development', 'code_comprehension',
            'integration_testing', 'security_analysis'
        ]
        
        # Try to find task category in the parts
        project_parts = []
        for i, part in enumerate(parts):
            # Check if this part or next part starts a task category
            if part in task_categories or (i < len(parts) - 1 and f"{part}_{parts[i+1]}" in task_categories):
                break
            project_parts.append(part)
        
        # If we didn't find a task category, take first 4 parts as fallback
        if not project_parts or len(project_parts) < 2:
            project_parts = parts[:4] if len(parts) >= 4 else parts
        
        return '_'.join(project_parts)
    
    def _get_code_file_path(self, scenario_id: str, context_file: str) -> Path:
        """Get full path to a code file from scenario ID and context file path.
        
        Args:
            scenario_id: Scenario ID (e.g., "c_api_gateway_easy_009_bug_investigation_expert_01")
            context_file: Relative path from context_files (e.g., "EduGate_ScholarLink//src//components//validator.c")
            
        Returns:
            Full Path object to the code file
        """
        # Extract project directory from scenario ID
        project_dir = self._extract_project_dir_from_id(scenario_id)
        
        # Build base path: {generated_dir}/{project_dir}
        generated_dir = Path(self.config.data.generated_dir)
        project_path = generated_dir / project_dir
        
        # Normalize context_file path (replace // with /)
        normalized_context_file = context_file.replace('//', '/')
        
        # Build full path
        full_path = project_path / normalized_context_file
        
        return full_path
    
    
    def select_relevant_content_from_files(
        self,
        scenario: Dict[str, Any],
        task_prompt: str,
        max_content_length: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Select relevant content from scenario's context files using LLM
        
        Args:
            scenario: Scenario dictionary with 'id' and 'context_files'
            task_prompt: The task prompt/description to guide content selection
            max_content_length: Optional maximum length for selected content per file
            
        Returns:
            Dictionary mapping file paths to selected relevant content
        """
        scenario_id = scenario.get('id', '')
        context_files_list = scenario.get('context_files', [])
        
        if not context_files_list:
            logger.debug(f"No context files for scenario {scenario_id}")
            return {}
        
        selected_content = {}
        
        # Load all context files
        file_contents = {}
        for context_file in context_files_list:
            try:
                file_path = self._get_code_file_path(scenario_id, context_file)
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_contents[context_file] = f.read()
                else:
                    logger.debug(f"Context file not found: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load context file {context_file}: {e}")
        
        if not file_contents:
            logger.warning(f"No context files loaded for scenario {scenario_id}")
            return {}
        
        # If LLM is available, use it to select relevant content
        if self.model and self.config.mcp_filter.use_llm_selection:
            try:
                # Use LLM to select relevant parts from each file
                for file_path, full_content in file_contents.items():
                    try:
                        # Skip LLM processing for very large files to avoid timeouts
                        if len(full_content) > self.max_file_size_for_llm:
                            logger.debug(f"File {file_path} too large ({len(full_content)} chars), using full content")
                            selected_content[file_path] = full_content
                            continue
                        
                        # Limit content size for prompt to avoid timeouts
                        content_for_prompt = full_content
                        if len(content_for_prompt) > self.max_prompt_size:
                            # Try to keep beginning and end of file (often most relevant)
                            half_size = self.max_prompt_size // 2
                            content_for_prompt = (
                                content_for_prompt[:half_size] + 
                                f"\n\n... [middle {len(full_content) - self.max_prompt_size} characters omitted] ...\n\n" +
                                content_for_prompt[-half_size:]
                            )
                            logger.debug(f"Truncated {file_path} from {len(full_content)} to {len(content_for_prompt)} chars for prompt")
                        
                        # Create prompt for content selection
                        selection_prompt = f"""Given the following task and code file, select the most relevant parts.

Task: {task_prompt}

Code file ({file_path}):
```
{content_for_prompt}
```

Select the most relevant code snippets, functions, classes, or sections that are directly related to the task. 
Return only the selected code, maintaining proper syntax and structure.
If the file is large, prioritize the parts that are most relevant to completing the task."""

                        # Get LLM response with retry logic
                        max_retries = 2
                        selected_code = None
                        last_error = None
                        
                        for attempt in range(max_retries + 1):
                            try:
                                response = self.model.invoke(selection_prompt)
                                selected_code = response.content if hasattr(response, 'content') else str(response)
                                
                                # Validate that we got meaningful content
                                if selected_code and len(selected_code.strip()) > 50:
                                    break
                                else:
                                    logger.warning(f"LLM returned empty/minimal content for {file_path}, attempt {attempt + 1}")
                                    if attempt < max_retries:
                                        continue
                                    # Fallback to full content if all retries failed
                                    selected_code = full_content
                                    break
                                    
                            except Exception as e:
                                last_error = e
                                if attempt < max_retries:
                                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s
                                    logger.debug(f"Retry {attempt + 1}/{max_retries} for {file_path} after {wait_time}s: {e}")
                                    time.sleep(wait_time)
                                else:
                                    # Final attempt failed
                                    raise
                        
                        if selected_code is None:
                            raise last_error or Exception("Failed to get response from LLM")
                        
                        # Apply length limit if specified
                        if max_content_length and len(selected_code) > max_content_length:
                            selected_code = selected_code[:max_content_length] + "\n... (truncated)"
                        
                        selected_content[file_path] = selected_code
                        logger.debug(f"Selected {len(selected_code)} chars from {file_path} (original: {len(full_content)} chars)")
                        
                    except Exception as e:
                        error_msg = str(e)
                        # Check if it's a timeout error
                        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                            logger.warning(f"Timeout selecting content from {file_path} ({len(full_content)} chars). Using full file.")
                        else:
                            logger.warning(f"Failed to select content from {file_path} using LLM: {e}. Using full file.")
                        selected_content[file_path] = full_content
                
            except Exception as e:
                logger.warning(f"LLM-based content selection failed: {e}. Using full file content.")
                selected_content = file_contents
        else:
            # No LLM available - return full file content
            selected_content = file_contents
            logger.debug(f"Using full file content for {len(file_contents)} files (LLM not available)")
        
        return selected_content


def create_scenario_filter(config: Config, base_url: str = None, api_key: str = None, model: str = None) -> LoCoBenchMCPTool:
    """
    Factory function to create an MCP tool instance
    
    Args:
        config: LoCoBench configuration object
        base_url: Base URL for the OpenAI-compatible API (defaults to config.mcp_filter.base_url)
        api_key: API key for authentication (defaults to config.mcp_filter.api_key)
        model: Model name for content selection (defaults to config.mcp_filter.model)
        
    Returns:
        LoCoBenchMCPTool instance
    """
    return LoCoBenchMCPTool(config, base_url=base_url, api_key=api_key, model=model)
