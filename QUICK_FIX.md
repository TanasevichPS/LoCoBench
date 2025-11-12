# Быстрое исправление ошибки MCP параметров

## Проблема

```
retrieve_relevant_embedding() got an unexpected keyword argument 'use_mcp'
```

## Решение

Параметры уже добавлены в функцию. Проблема может быть в кэшировании Python.

### Шаг 1: Очистить кэш (уже сделано)

```bash
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

### Шаг 2: Проверить файл

Убедитесь, что в `locobench/retrieval.py` функция `retrieve_relevant_embedding` имеет параметры:

```python
def retrieve_relevant_embedding(
    context_files: Optional[Dict[str, str]],
    task_prompt: str,
    *,
    # ... другие параметры ...
    task_category: Optional[str] = None,
    use_mcp: bool = False,  # ← Должен быть
    mcp_provider: Optional[str] = None,  # ← Должен быть
    mcp_model: Optional[str] = None,  # ← Должен быть
    mcp_base_url: Optional[str] = None,  # ← Должен быть
    mcp_api_key: Optional[str] = None,  # ← Должен быть
    config: Optional[Any] = None,  # ← Должен быть
) -> str:
```

### Шаг 3: Перезапустить

Перезапустите Python процесс полностью (если запущен в фоне).

### Шаг 4: Попробовать снова

```bash
python -m locobench.cli evaluate \
    --scenarios data/output/scenarios/test_easy_scenario.json \
    --config config.yaml
```

## Если проблема сохраняется

Проверьте строку 1298-1303 в `locobench/retrieval.py` - там должны быть все параметры MCP.

Все параметры имеют дефолтные значения (`= False` или `= None`), так что они опциональны и не должны вызывать ошибку.
