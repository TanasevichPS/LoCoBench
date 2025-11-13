# MCP Scenario Filter Tool

This module provides an MCP (Model Context Protocol) tool for LoCoBench that uses ChatOpenAI with AgentExecutor to intelligently filter and select relevant scenarios from files.

## Features

- **LLM-based Selection**: Uses ChatOpenAI with AgentExecutor to intelligently select relevant scenarios
- **Filtering by Difficulty**: Filter scenarios by difficulty levels (easy, medium, hard, expert)
- **Filtering by Language**: Filter scenarios by supported programming languages
- **Filtering by Task Category**: Filter scenarios by task categories
- **File Reading**: Reads scenario JSON files from a directory
- **Agent Tools**: Provides tools for the agent to read files, extract metadata, and filter scenarios

## Installation

Make sure you have the required dependencies installed:

```bash
pip install langchain langchain-openai langchain-core
```

## Usage

### Basic Usage

```python
from pathlib import Path
from locobench.core.config import Config
from locobench.mcp_scenario_filter import create_scenario_filter

# Load configuration
config = Config.from_yaml("config.yaml")

# Create scenario filter
scenario_filter = create_scenario_filter(
    config,
    base_url="http://localhost:8000/v1",  # Your OpenAI-compatible API endpoint
    api_key="your-api-key"  # Your API key
)

# Filter scenarios
scenarios_dir = Path(config.data.output_dir) / "scenarios"
filtered_scenarios = scenario_filter.filter_scenarios_from_files(
    scenarios_dir=scenarios_dir,
    difficulty_levels=["easy", "medium"],
    task_categories=None,  # None means all categories
    use_llm_selection=True  # Enable LLM-based selection
)

print(f"Filtered {len(filtered_scenarios)} scenarios")
```

### Advanced Usage

```python
# Filter with specific criteria
filtered_scenarios = scenario_filter.filter_scenarios_from_files(
    scenarios_dir=scenarios_dir,
    difficulty_levels=["hard", "expert"],
    task_categories=["architectural_understanding", "bug_investigation"],
    use_llm_selection=True
)

# Access filtered scenarios
for scenario in filtered_scenarios:
    print(f"ID: {scenario.get('id')}")
    print(f"Difficulty: {scenario.get('difficulty')}")
    print(f"Language: {scenario_filter._get_scenario_language(scenario)}")
    print(f"Task Category: {scenario.get('task_category')}")
```

## Configuration

The tool uses the following ChatOpenAI configuration:

- **model**: "gpt-oss"
- **temperature**: 0.0 (deterministic)
- **base_url**: Configurable (default: "http://localhost:8000/v1")
- **api_key**: Configurable (default: "111")
- **streaming**: True
- **timeout**: 30.0 seconds

## Agent Tools

The agent has access to the following tools:

1. **read_scenario_file**: Reads a scenario JSON file and returns its contents
2. **read_code_file**: Reads a code file referenced in a scenario's context_files
   - Requires: `scenario_id` and `context_file` path
   - Resolves path: `{generated_dir}/{project_dir}/{context_file}`
   - Example: `read_code_file("c_api_gateway_easy_009_bug_investigation_expert_01", "EduGate_ScholarLink//src//components//validator.c")`
3. **read_scenario_context_files**: Reads all context files for a scenario
   - Reads up to `max_files` (default: 10) context files
   - Returns file contents with metadata
4. **get_scenario_metadata**: Extracts metadata from a scenario
   - Returns: id, difficulty, language, task_category, project_dir, context_files_count
5. **filter_by_difficulty**: Filters scenarios by difficulty level
6. **filter_by_language**: Filters scenarios by programming language

## Scenario ID Structure

Scenario IDs follow this pattern:
```
{language}_{project_name}_{complexity}_{number}_{task_category}_{difficulty}_{instance}
```

Example: `c_api_gateway_easy_009_bug_investigation_expert_01`

- **Language**: `c` (extracted for language filtering)
- **Project Directory**: `c_api_gateway_easy_009` (used to locate code files)
- **Task Category**: `bug_investigation`
- **Difficulty**: `expert`
- **Instance**: `01`

## Code File Path Resolution

The tool automatically resolves code file paths using:

1. **Scenario ID** → Extracts project directory name
2. **Config** → Uses `config.data.generated_dir` as base path
3. **Context File** → From scenario's `context_files` array

Full path formula:
```
{generated_dir}/{project_dir}/{context_file}
```

Example:
- Scenario ID: `c_api_gateway_easy_009_bug_investigation_expert_01`
- Generated Dir: `/srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/data/generated`
- Project Dir: `c_api_gateway_easy_009` (extracted from ID)
- Context File: `EduGate_ScholarLink//src//components//validator.c`
- Full Path: `/srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/data/generated/c_api_gateway_easy_009/EduGate_ScholarLink/src/components/validator.c`

## Filtering Logic

The tool implements filtering similar to the evaluator's `_filter_scenarios` method:

1. **Language Filtering**: Filters scenarios based on `config.phase1.supported_languages`
2. **Difficulty Filtering**: Filters scenarios by specified difficulty levels
3. **Task Category Filtering**: Filters scenarios by specified task categories

The `_get_scenario_language` method extracts the language from the scenario ID using a mapping:
- `c`, `cpp`, `cs`/`csharp`, `go`, `java`, `js`/`javascript`, `php`, `py`/`python`, `rs`/`rust`, `ts`/`typescript`

## Example

See `mcp_example.py` for a complete usage example.

## Notes

- The tool reads scenario files from the `scenarios` directory (typically `./data/output/scenarios`)
- Each scenario file contains a single scenario object (not an array)
- The LLM agent can intelligently select scenarios based on the filtering criteria
- If LLM selection fails, the tool falls back to basic filtering
