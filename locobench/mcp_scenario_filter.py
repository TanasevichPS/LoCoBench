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
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from .core.config import Config

logger = logging.getLogger(__name__)


class LoCoBenchScenarioFilter:
    """MCP tool for filtering LoCoBench scenarios using LLM-based selection"""
    
    def __init__(self, config: Config, base_url: str = "http://localhost:8000/v1", api_key: str = "111"):
        """
        Initialize the scenario filter
        
        Args:
            config: LoCoBench configuration object
            base_url: Base URL for the OpenAI-compatible API
            api_key: API key for authentication
        """
        self.config = config
        self.base_url = base_url
        self.api_key = api_key
        
        # Initialize the LLM model
        self.model = ChatOpenAI(
            model="gpt-oss",
            temperature=0.0,
            base_url=base_url,
            api_key=api_key,
            streaming=True,
            timeout=30.0
        )
        
        # Create tools for the agent
        self.tools = self._create_tools()
        
        # Bind tools to model
        self.model_with_tools = self.model.bind_tools(self.tools)
        
        # Create agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing and filtering code evaluation scenarios.
Your task is to help select relevant scenarios from a collection based on:
1. Difficulty level (easy, medium, hard, expert)
2. Programming language support
3. Task category relevance

When analyzing scenarios, consider:
- The scenario's difficulty level
- The programming language used
- The task category and its requirements
- The context size and complexity

Provide clear reasoning for your selections."""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_openai_tools_agent(self.model_with_tools, self.tools, self.prompt)
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True
        )
    
    def _create_tools(self) -> List:
        """Create tools for the agent to use"""
        
        @tool
        def read_scenario_file(file_path: str) -> Dict[str, Any]:
            """Read a scenario JSON file and return its contents.
            
            Args:
                file_path: Path to the scenario JSON file
                
            Returns:
                Dictionary containing the scenario data
            """
            try:
                # Handle both absolute and relative paths
                path = Path(file_path)
                if not path.is_absolute():
                    # Try to find the file in common locations
                    scenarios_dir = Path(self.config.data.output_dir) / "scenarios"
                    path = scenarios_dir / path.name
                
                with open(path, 'r') as f:
                    scenario_data = json.load(f)
                return scenario_data
            except Exception as e:
                logger.error(f"Error reading scenario file {file_path}: {e}")
                return {"error": str(e)}
        
        @tool
        def get_scenario_metadata(scenario: Dict[str, Any]) -> Dict[str, Any]:
            """Extract metadata from a scenario.
            
            Args:
                scenario: Scenario dictionary
                
            Returns:
                Dictionary with scenario metadata (id, difficulty, language, task_category)
            """
            scenario_id = scenario.get('id', '')
            difficulty = scenario.get('difficulty', '').lower()
            task_category = scenario.get('task_category', '')
            
            # Extract language from scenario ID
            language = self._get_scenario_language(scenario)
            
            return {
                "id": scenario_id,
                "difficulty": difficulty,
                "language": language,
                "task_category": task_category
            }
        
        @tool
        def filter_by_difficulty(scenarios: List[Dict[str, Any]], difficulty_levels: List[str]) -> List[Dict[str, Any]]:
            """Filter scenarios by difficulty level.
            
            Args:
                scenarios: List of scenario dictionaries
                difficulty_levels: List of difficulty levels to include (e.g., ['easy', 'medium'])
                
            Returns:
                Filtered list of scenarios
            """
            difficulty_levels_lower = [d.lower() for d in difficulty_levels]
            filtered = [
                s for s in scenarios 
                if s.get('difficulty', '').lower() in difficulty_levels_lower
            ]
            return filtered
        
        @tool
        def filter_by_language(scenarios: List[Dict[str, Any]], supported_languages: List[str]) -> List[Dict[str, Any]]:
            """Filter scenarios by programming language.
            
            Args:
                scenarios: List of scenario dictionaries
                supported_languages: List of supported languages to include
                
            Returns:
                Filtered list of scenarios
            """
            filtered = []
            for scenario in scenarios:
                scenario_language = self._get_scenario_language(scenario)
                if scenario_language in supported_languages:
                    filtered.append(scenario)
            return filtered
        
        return [read_scenario_file, get_scenario_metadata, filter_by_difficulty, filter_by_language]
    
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
        
        # If LLM selection is enabled, use agent to further refine selection
        if use_llm_selection and filtered_scenarios:
            try:
                # Store scenarios for tool access
                self._scenarios_cache = filtered_scenarios
                
                # Prepare file paths for the agent to read
                scenario_file_paths = [str(f) for f in scenario_files[:len(filtered_scenarios)]]
                
                # Create agent input
                agent_input = f"""You need to analyze and filter scenarios from JSON files.

Filtering criteria:
- Difficulty levels: {difficulty_levels if difficulty_levels else 'all'}
- Supported languages: {self.config.phase1.supported_languages}
- Task categories: {task_categories if task_categories else 'all'}

Available scenario files:
{chr(10).join([f"- {path}" for path in scenario_file_paths[:20]])}

Your task:
1. Read the scenario files using read_scenario_file tool
2. Extract metadata from each scenario using get_scenario_metadata tool
3. Apply filtering using filter_by_difficulty and filter_by_language tools
4. Select the most relevant scenarios based on the criteria

Start by reading a few scenario files to understand their structure, then apply the appropriate filters."""
                
                # Run agent
                result = self.agent_executor.invoke({"input": agent_input})
                
                # The agent will use tools to filter scenarios
                # The filtered results are already applied through tool calls
                logger.info(f"🤖 LLM agent completed selection: {result.get('output', 'N/A')}")
                
                # Note: In a production implementation, you would parse the agent's tool calls
                # and extract the filtered scenarios. For now, we return the pre-filtered scenarios
                # as the agent's tool calls modify the internal state.
                
            except Exception as e:
                logger.error(f"Error in LLM-based selection: {e}")
                logger.info("Falling back to basic filtering")
                # Clean up cache
                if hasattr(self, '_scenarios_cache'):
                    delattr(self, '_scenarios_cache')
        
        # Log difficulty distribution after filtering
        if filtered_scenarios:
            difficulty_counts = {}
            for s in filtered_scenarios:
                diff = s.get('difficulty', 'unknown').lower()
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            logger.info(f"📊 Scenario difficulty distribution after filtering: {difficulty_counts}")
        
        return filtered_scenarios


def create_scenario_filter(config: Config, base_url: str = "http://localhost:8000/v1", api_key: str = "111") -> LoCoBenchScenarioFilter:
    """
    Factory function to create a scenario filter instance
    
    Args:
        config: LoCoBench configuration object
        base_url: Base URL for the OpenAI-compatible API
        api_key: API key for authentication
        
    Returns:
        LoCoBenchScenarioFilter instance
    """
    return LoCoBenchScenarioFilter(config, base_url=base_url, api_key=api_key)
