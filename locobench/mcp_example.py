"""
Example usage of the MCP Scenario Filter tool

This demonstrates how to use the LoCoBenchScenarioFilter to filter scenarios
using LLM-based selection with ChatOpenAI and AgentExecutor.
"""

import asyncio
from pathlib import Path
from locobench.core.config import Config
from locobench.mcp_scenario_filter import create_scenario_filter


async def main():
    """Example usage of the MCP scenario filter"""
    
    # Load configuration
    config = Config.from_yaml("config.yaml")
    
    # Create scenario filter with custom API settings
    # Note: Update base_url and api_key according to your setup
    scenario_filter = create_scenario_filter(
        config,
        base_url="http://localhost:8000/v1",  # Update with your API endpoint
        api_key="111"  # Update with your API key
    )
    
    # Define scenarios directory
    scenarios_dir = Path(config.data.output_dir) / "scenarios"
    
    # Filter scenarios by difficulty and language
    filtered_scenarios = scenario_filter.filter_scenarios_from_files(
        scenarios_dir=scenarios_dir,
        difficulty_levels=["easy", "medium"],  # Filter by difficulty
        task_categories=None,  # None means all categories
        use_llm_selection=True  # Enable LLM-based intelligent selection
    )
    
    print(f"\n✅ Filtered {len(filtered_scenarios)} scenarios")
    print("\nFiltered scenario IDs:")
    for scenario in filtered_scenarios[:10]:  # Show first 10
        scenario_id = scenario.get('id', 'unknown')
        difficulty = scenario.get('difficulty', 'unknown')
        language = scenario_filter._get_scenario_language(scenario)
        project_dir = scenario_filter._extract_project_dir_from_id(scenario_id)
        context_files_count = len(scenario.get('context_files', []))
        
        print(f"  - {scenario_id}")
        print(f"    Difficulty: {difficulty}, Language: {language}")
        print(f"    Project Dir: {project_dir}, Context Files: {context_files_count}")
    
    # Example: Read a code file from a scenario
    if filtered_scenarios:
        example_scenario = filtered_scenarios[0]
        scenario_id = example_scenario.get('id')
        context_files = example_scenario.get('context_files', [])
        
        if context_files:
            print(f"\n📄 Example: Reading code file from scenario {scenario_id}")
            print(f"   Context file: {context_files[0]}")
            
            # Get the full path
            full_path = scenario_filter._get_code_file_path(scenario_id, context_files[0])
            print(f"   Full path: {full_path}")
            
            # Check if file exists
            if full_path.exists():
                print(f"   ✅ File exists")
                # You can read it using: read_code_file tool or directly
            else:
                print(f"   ❌ File not found")
    
    return filtered_scenarios


if __name__ == "__main__":
    asyncio.run(main())
