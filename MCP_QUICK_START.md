# Быстрый старт: Интеграция MCP с LLM

## Что было создано

1. **`locobench/mcp_retrieval.py`** - MCP Server с tools для разных типов задач
2. **`locobench/mcp_llm_integration.py`** - Интеграция с OpenAI и Anthropic API
3. **`test_mcp_integration.py`** - Тестовый скрипт
4. **Документация** - Подробные руководства

## Как использовать

### Простой пример (без LLM - эвристики)

```python
from pathlib import Path
from locobench.mcp_retrieval import retrieve_with_mcp

result = retrieve_with_mcp(
    context_files={"file.py": "code..."},
    task_prompt="Найти уязвимости безопасности",
    task_category="security_analysis",
    project_dir=Path("."),
    use_llm=False,  # Использовать эвристики
)
```

### С OpenAI

```python
from locobench.mcp_retrieval import retrieve_with_mcp
from locobench.core.config import Config

config = Config.from_yaml("config.yaml")

result = retrieve_with_mcp(
    context_files=context_files,
    task_prompt="Найти уязвимости безопасности",
    task_category="security_analysis",
    project_dir=Path("."),
    config=config,
    provider="openai",
    model="gpt-4o",
    use_llm=True,
)
```

### С Anthropic (Claude)

```python
result = retrieve_with_mcp(
    context_files=context_files,
    task_prompt="Найти уязвимости безопасности",
    task_category="security_analysis",
    project_dir=Path("."),
    config=config,
    provider="anthropic",
    model="claude-sonnet-4",
    use_llm=True,
)
```

## Настройка API ключей

В `config.yaml`:
```yaml
api:
  openai_api_key: "sk-..."
  claude_bearer_token: "sk-ant-..."
```

Или через переменные окружения:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."  # или CLAUDE_BEARER_TOKEN
```

## Тестирование

```bash
python test_mcp_integration.py
```

## Интеграция в существующий код

В `locobench/retrieval.py` добавьте параметр `use_mcp`:

```python
result = retrieve_relevant_embedding(
    ...,
    task_category=task.category.value,
    use_mcp=True,  # Включить MCP
    mcp_provider="openai",
    mcp_model="gpt-4o",
    config=config,
)
```

## Поддерживаемые типы задач

- `security_analysis` - Анализ безопасности
- `architectural_understanding` - Понимание архитектуры
- `code_comprehension` - Понимание кода
- `feature_implementation` - Реализация функций
- `bug_investigation` - Расследование багов
- `cross_file_refactoring` - Рефакторинг между файлами
- `integration_testing` - Интеграционное тестирование
- `multi_session_development` - Многосессионная разработка

## Как это работает

1. MCP Server создает специализированные tools для типа задачи
2. LLM анализирует задачу и вызывает нужные tools
3. Tools выполняются и возвращают релевантные файлы
4. LLM может запросить дополнительные файлы
5. Формируется финальный контекст

## Дополнительная информация

- `MCP_INTEGRATION_GUIDE.md` - Подробное руководство
- `MCP_EXPLANATION_RU.md` - Объяснение концепций
- `MCP_INTEGRATION_PLAN.md` - План интеграции
