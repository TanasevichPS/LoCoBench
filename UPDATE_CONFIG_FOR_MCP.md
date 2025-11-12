# Как обновить config.yaml для использования MCP с эвристиками

## Быстрое обновление

Добавьте следующие строки в секцию `retrieval:` вашего `config.yaml`:

```yaml
retrieval:
  # ... существующие параметры ...
  
  # Включить MCP с эвристиками (без LLM)
  use_mcp: true
  mcp_provider: "heuristics"  # или просто не указывайте - будет использоваться автоматически
```

## Полный пример секции retrieval

```yaml
retrieval:
  chunk_size: 512
  difficulties:
    - hard
    - expert
  enabled: true
  top_percent: 0.22
  max_context_tokens: 100000
  method: embedding
  model_name: all-MiniLM-L6-v2
  local_model_path: null
  top_k: 10
  smart_chunking: true
  chunks_per_file: 8
  retrieval_chunk_size: 2500
  use_multi_query: true
  use_hybrid_search: true
  hybrid_alpha: 0.75
  
  # MCP с эвристиками (добавьте эти строки)
  use_mcp: true
  mcp_provider: "heuristics"  # или оставьте пустым
```

## Минимальное изменение

Если хотите минимальное изменение, просто добавьте одну строку:

```yaml
retrieval:
  # ... все существующие параметры ...
  use_mcp: true  # Добавьте только эту строку
```

Система автоматически определит, что нужно использовать эвристики.

## Проверка

После обновления config.yaml запустите:

```bash
python -m locobench.cli evaluate \
    --scenarios data/output/scenarios/test_easy_scenario.json \
    --config config.yaml
```

В логах вы должны увидеть:
```
INFO: 🔧 Using MCP-based retrieval (provider=heuristics, use_llm=False)
INFO: 📋 Created MCP server with X tools
INFO: ✅ MCP heuristics retrieval returned X characters
```

## Отключение MCP

Если нужно временно отключить MCP:

```yaml
retrieval:
  use_mcp: false  # или просто удалите строку
```

Система вернется к стандартному retrieval.
