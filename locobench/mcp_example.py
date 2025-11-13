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
        print(f"  - {scenario.get('id', 'unknown')} ({scenario.get('difficulty', 'unknown')})")
    
    return filtered_scenarios


if __name__ == "__main__":
    asyncio.run(main())
