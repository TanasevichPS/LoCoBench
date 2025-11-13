"""
Example usage of the MCP Tool

This demonstrates how to use the LoCoBenchMCPTool to select relevant content
from code files for scenarios using LLM-based content selection.
"""

import asyncio
import json
from pathlib import Path
from locobench.core.config import Config
from locobench.mcp_scenario_filter import create_scenario_filter


async def main():
    """Example usage of the MCP tool for content selection"""
    
    # Load configuration
    config = Config.from_yaml("config.yaml")
    
    # Create MCP tool with custom API settings
    # Note: Update base_url and api_key according to your setup
    mcp_tool = create_scenario_filter(
        config,
        base_url="http://localhost:8000/v1",  # Update with your API endpoint
        api_key="111"  # Update with your API key
    )
    
    # Load a sample scenario
    scenarios_dir = Path(config.data.output_dir) / "scenarios"
    scenario_files = list(scenarios_dir.glob("*.json"))
    
    if not scenario_files:
        print("No scenario files found!")
        return
    
    # Load first scenario
    with open(scenario_files[0], 'r') as f:
        scenario = json.load(f)
    
    scenario_id = scenario.get('id', 'unknown')
    task_prompt = scenario.get('task_prompt', '')
    context_files = scenario.get('context_files', [])
    
    print(f"\n📋 Scenario: {scenario_id}")
    print(f"   Task: {task_prompt[:100]}...")
    print(f"   Context files: {len(context_files)}")
    
    # Example: Select relevant content from context files
    if context_files and config.mcp_filter.enabled:
        print(f"\n🔍 Using MCP tool to select relevant content from files...")
        selected_content = mcp_tool.select_relevant_content_from_files(
            scenario=scenario,
            task_prompt=task_prompt,
            max_content_length=None
        )
        
        print(f"\n✅ Selected content from {len(selected_content)} files:")
        for file_path, content in selected_content.items():
            print(f"   - {file_path}: {len(content)} characters")
            print(f"     Preview: {content[:100]}...")
    else:
        print("\n⚠️  MCP tool disabled or no context files. Enable it in config.yaml")
    
    # Example: Get file path resolution
    if context_files:
        print(f"\n📄 Example: Resolving file paths")
        for context_file in context_files[:3]:  # Show first 3
            full_path = mcp_tool._get_code_file_path(scenario_id, context_file)
            print(f"   {context_file}")
            print(f"   → {full_path}")
            print(f"   Exists: {full_path.exists()}")
    
    return scenario


if __name__ == "__main__":
    asyncio.run(main())
