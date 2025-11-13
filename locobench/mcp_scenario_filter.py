"""
MCP Tool for LoCoBench Scenario Filtering

This module provides an MCP tool that uses ChatOpenAI with AgentExecutor
to intelligently select relevant scenarios from files based on difficulty
and supported languages.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .core.config import Config

logger = logging.getLogger(__name__)


class LoCoBenchScenarioFilter:
    """MCP tool for filtering LoCoBench scenarios using LLM-based selection"""
    
    def __init__(self, config: Config, base_url: str = None, api_key: str = None, model: str = None):
        """
        Initialize the scenario filter
        
        Args:
            config: LoCoBench configuration object
            base_url: Base URL for the OpenAI-compatible API (defaults to config.mcp_filter.base_url)
            api_key: API key for authentication (defaults to config.mcp_filter.api_key)
            model: Model name for the filter agent (defaults to config.mcp_filter.model)
        """
        self.config = config
        
        # Use config values as defaults
        self.base_url = base_url or config.mcp_filter.base_url
        self.api_key = api_key or config.mcp_filter.api_key
        model_name = model or config.mcp_filter.model
        
        # Note: LangChain agent functionality removed - using basic filtering only
        # The MCP tool now focuses on file path resolution and basic filtering
        self.agent_executor = None
        self.model = None
        self.tools = []
        
        if config.mcp_filter.use_llm_selection:
            logger.warning("LLM-based selection requested but LangChain is not available. Using basic filtering only.")
    
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
    
    def _create_tools(self) -> List:
        """Create tools for the agent to use - NOT USED (LangChain removed)"""
        # This method is kept for compatibility but returns empty list
        # LangChain agent functionality has been removed - we only use basic filtering now
        return []
    
    def _get_scenario_language(self, scenario: Dict[str, Any]) -> str:
        """Extract language from scenario ID"""
        scenario_id = scenario.get('id', '')
        if not scenario_id:
            return 'unknown'

        parts = scenario_id.split('_')
        if not parts:
            return 'unknown'
        
        language = parts[0].lower()

        language_mapping = {
            'c': 'c',
            'cpp': 'cpp', 
            'cs': 'csharp', 'csharp': 'csharp',
            'go': 'go',
            'java': 'java',
            'js': 'javascript', 'javascript': 'javascript',
            'php': 'php',
            'py': 'python', 'python': 'python',
            'rs': 'rust', 'rust': 'rust',
            'ts': 'typescript', 'typescript': 'typescript'
        }
        
        return language_mapping.get(language, language)
    
    def _filter_scenarios(self, scenarios: List[Dict[str, Any]], 
                         task_categories: Optional[List[str]], 
                         difficulty_levels: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Filter scenarios based on criteria"""
        
        filtered = scenarios
        supported_languages = self.config.phase1.supported_languages
        
        # Filter by language if supported_languages is configured
        if supported_languages:
            filtered = [s for s in filtered if self._get_scenario_language(s) in supported_languages]
            logger.info(f"🌍 Language filtering: {len(scenarios)} → {len(filtered)} scenarios")
        
        if task_categories:
            filtered = [s for s in filtered if s.get('task_category') in task_categories]
        
        if difficulty_levels:
            difficulty_levels_lower = [d.lower() for d in difficulty_levels]
            filtered = [s for s in filtered if s.get('difficulty', '').lower() in difficulty_levels_lower]
        
        return filtered
    
    def filter_scenarios_from_files(
        self,
        scenarios_dir: Path,
        difficulty_levels: Optional[List[str]] = None,
        task_categories: Optional[List[str]] = None,
        use_llm_selection: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Read scenarios from files and filter them using LLM-based selection
        
        Args:
            scenarios_dir: Directory containing scenario JSON files
            difficulty_levels: Optional list of difficulty levels to filter by
            task_categories: Optional list of task categories to filter by
            use_llm_selection: Whether to use LLM agent for intelligent selection
            
        Returns:
            List of filtered scenario dictionaries
        """
        # Load all scenarios from files
        all_scenarios = []
        scenario_files = list(scenarios_dir.glob("*.json"))
        
        logger.info(f"📁 Found {len(scenario_files)} scenario files in {scenarios_dir}")
        
        for scenario_file in scenario_files:
            try:
                with open(scenario_file, 'r') as f:
                    scenario_data = json.load(f)
                    # Each file contains a single scenario object
                    all_scenarios.append(scenario_data)
            except Exception as e:
                logger.error(f"Error loading scenario from {scenario_file}: {e}")
                continue
        
        if not all_scenarios:
            logger.warning("No scenarios found in scenario files!")
            return []
        
        logger.info(f"📊 Loaded {len(all_scenarios)} scenarios")
        
        # Apply basic filtering first
        filtered_scenarios = self._filter_scenarios(
            all_scenarios,
            task_categories=task_categories,
            difficulty_levels=difficulty_levels
        )
        
        logger.info(f"🔍 After basic filtering: {len(all_scenarios)} → {len(filtered_scenarios)} scenarios")
        
        # LLM-based selection removed - using basic filtering only
        if use_llm_selection:
            logger.info("LLM-based selection requested but LangChain agent functionality has been removed. Using basic filtering only.")
        
        # Log difficulty distribution after filtering
        if filtered_scenarios:
            difficulty_counts = {}
            for s in filtered_scenarios:
                diff = s.get('difficulty', 'unknown').lower()
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            logger.info(f"📊 Scenario difficulty distribution after filtering: {difficulty_counts}")
        
        return filtered_scenarios


def create_scenario_filter(config: Config, base_url: str = None, api_key: str = None, model: str = None) -> LoCoBenchScenarioFilter:
    """
    Factory function to create a scenario filter instance
    
    Args:
        config: LoCoBench configuration object
        base_url: Base URL for the OpenAI-compatible API (defaults to config.mcp_filter.base_url)
        api_key: API key for authentication (defaults to config.mcp_filter.api_key)
        model: Model name for the filter agent (defaults to config.mcp_filter.model)
        
    Returns:
        LoCoBenchScenarioFilter instance
    """
    return LoCoBenchScenarioFilter(config, base_url=base_url, api_key=api_key, model=model)
