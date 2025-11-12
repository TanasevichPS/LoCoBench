# Исправление ошибки с параметрами MCP

## Проблема

Ошибка: `retrieve_relevant_embedding() got an unexpected keyword argument 'use_mcp'`

## Причина

Параметры MCP были добавлены в функцию `retrieve_relevant_embedding`, но возможно:
1. Python кэширует старую версию модуля
2. Где-то функция вызывается напрямую без обновления

## Решение

### Шаг 1: Очистить кэш Python

```bash
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

### Шаг 2: Проверить, что параметры добавлены

Параметры должны быть в функции `retrieve_relevant_embedding`:

```python
def retrieve_relevant_embedding(
    context_files: Optional[Dict[str, str]],
    task_prompt: str,
    *,
    # ... другие параметры ...
    use_mcp: bool = False,  # Use MCP-based retrieval
    mcp_provider: Optional[str] = None,
    mcp_model: Optional[str] = None,
    mcp_base_url: Optional[str] = None,
    mcp_api_key: Optional[str] = None,
    config: Optional[Any] = None,
) -> str:
```

### Шаг 3: Перезапустить Python

После очистки кэша перезапустите Python процесс.

## Проверка

Запустите снова:

```bash
python -m locobench.cli evaluate \
    --scenarios data/output/scenarios/test_easy_scenario.json \
    --config config.yaml
```

## Если проблема сохраняется

Проверьте, что все параметры имеют дефолтные значения (они есть - все `= False` или `= None`).

Также убедитесь, что вы используете обновленную версию файла `retrieval.py`.
