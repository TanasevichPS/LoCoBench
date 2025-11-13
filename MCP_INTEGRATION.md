# MCP Scenario Filter Integration

The MCP Scenario Filter has been integrated into the LoCoBench evaluation system, allowing intelligent scenario filtering using LLM-based selection.

## Usage

### Via Configuration File

Enable MCP filtering in your `config.yaml`:

```yaml
mcp_filter:
  enabled: true
  base_url: "http://localhost:8000/v1"
  api_key: "your-api-key"
  model: "gpt-oss"
  use_llm_selection: true
```

Then run:
```bash
locobench evaluate --model "DeepSeekR1-70B-LRI" --config-path full_java_config.yaml
```

### Via Command Line

Override config settings from command line:
```bash
locobench evaluate \
  --model "DeepSeekR1-70B-LRI" \
  --config-path full_java_config.yaml \
  --enable-mcp-filter \
  --mcp-base-url "http://localhost:8000/v1" \
  --mcp-api-key "your-api-key"
```

## How It Works

1. **Scenario Loading**: All scenarios are loaded from the `scenarios` directory
2. **MCP Filtering**: If enabled, the MCP filter tool:
   - Uses ChatOpenAI with AgentExecutor to intelligently select scenarios
   - Filters by difficulty, language, and task category
   - Can read actual code files to make informed decisions
3. **Evaluation**: Only filtered scenarios are evaluated

## Configuration Options

- `enabled`: Enable/disable MCP filtering (default: false)
- `base_url`: Base URL for OpenAI-compatible API (default: "http://localhost:8000/v1")
- `api_key`: API key for authentication (default: "111")
- `model`: Model name for filter agent (default: "gpt-oss")
- `use_llm_selection`: Use LLM-based intelligent selection vs basic filtering (default: true)

## Features

- **Intelligent Selection**: LLM agent analyzes scenarios and selects relevant ones
- **Code File Reading**: Agent can read actual code files from scenarios
- **Multi-criteria Filtering**: Filters by difficulty, language, and task category
- **Fallback**: If MCP filter fails, falls back to using all scenarios

## Example Config File

Create `full_java_config.yaml`:

```yaml
api:
  max_requests_per_minute: 600
  max_concurrent_requests: 300

data:
  output_dir: "./data/output"
  generated_dir: "./data/generated"

phase1:
  supported_languages:
    - java
  projects_per_language: 100

phase3:
  total_instances: 8000
  task_distribution:
    architectural_understanding: 1000
    cross_file_refactoring: 1000
    feature_implementation: 1000
    bug_investigation: 1000
    multi_session_development: 1000
    code_comprehension: 1000
    integration_testing: 1000
    security_analysis: 1000
  difficulty_distribution:
    easy: 2000
    medium: 2000
    hard: 2000
    expert: 2000

mcp_filter:
  enabled: true
  base_url: "http://localhost:8000/v1"
  api_key: "your-api-key"
  model: "gpt-oss"
  use_llm_selection: true
```

Then run:
```bash
locobench evaluate --model "DeepSeekR1-70B-LRI" --config-path full_java_config.yaml
```
