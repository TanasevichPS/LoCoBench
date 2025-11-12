# Руководство по интеграции MCP Tools с LLM

## Обзор

Это руководство объясняет, как интегрировать MCP (Model Context Protocol) tools с LLM клиентами (OpenAI и Anthropic) для интеллектуального выбора файлов в LoCoBench.

## Архитектура

```
Task → MCP Server → LLM Client → Tool Calls → File Selection → Context
```

1. **MCP Server** создает специализированные tools для типа задачи
2. **LLM Client** (OpenAI/Anthropic) анализирует задачу и вызывает tools
3. **Tools** выполняются и возвращают релевантные файлы
4. **LLM** может запросить дополнительные файлы через новые tool calls
5. **Финальный контекст** формируется из выбранных файлов

## Установка зависимостей

Убедитесь, что установлены необходимые библиотеки:

```bash
pip install openai>=1.0.0 anthropic>=0.7.0
```

## Базовое использование

### Пример 1: Использование с OpenAI

```python
from pathlib import Path
from locobench.mcp_retrieval import retrieve_with_mcp
from locobench.core.config import Config

# Загрузка конфигурации
config = Config.from_yaml("config.yaml")

# Контекстные файлы проекта
context_files = {
    "src/auth.py": "...",
    "src/security.py": "...",
    "src/api.py": "...",
}

# Задача
task_prompt = "Найти уязвимости в обработке пользовательского ввода"
task_category = "security_analysis"
project_dir = Path("/path/to/project")

# Retrieval через MCP с OpenAI
result = retrieve_with_mcp(
    context_files=context_files,
    task_prompt=task_prompt,
    task_category=task_category,
    project_dir=project_dir,
    config=config,
    provider="openai",
    model="gpt-4o",  # или "o3" для reasoning модели
    use_llm=True,
)

print(result)  # Форматированный контекст с выбранными файлами
```

### Пример 2: Использование с Anthropic (Claude)

```python
from locobench.mcp_retrieval import retrieve_with_mcp
from locobench.core.config import Config

config = Config.from_yaml("config.yaml")

result = retrieve_with_mcp(
    context_files=context_files,
    task_prompt=task_prompt,
    task_category=task_category,
    project_dir=project_dir,
    config=config,
    provider="anthropic",  # или "claude"
    model="claude-sonnet-4",
    use_llm=True,
)
```

### Пример 3: Прямое использование MCPLLMIntegrator

```python
import asyncio
from locobench.mcp_retrieval import LoCoBenchMCPServer
from locobench.mcp_llm_integration import MCPLLMIntegrator
from locobench.core.config import Config

async def main():
    # Создание MCP сервера
    mcp_server = LoCoBenchMCPServer(
        project_dir=Path("/path/to/project"),
        task_category="security_analysis",
        context_files=context_files,
        task_prompt="Найти уязвимости безопасности",
    )
    
    # Создание интегратора
    config = Config.from_yaml("config.yaml")
    integrator = MCPLLMIntegrator(
        mcp_server=mcp_server,
        config=config,
    )
    
    # Retrieval с OpenAI
    result_openai = await integrator.retrieve_with_openai(
        model="gpt-4o",
        max_iterations=5,
    )
    
    # Или с Anthropic
    result_anthropic = await integrator.retrieve_with_anthropic(
        model="claude-sonnet-4",
        max_iterations=5,
    )
    
    print(result_openai)

asyncio.run(main())
```

## Интеграция с существующим кодом

### Обновление `retrieve_relevant_embedding`

Добавьте поддержку MCP в существующую функцию retrieval:

```python
# В locobench/retrieval.py

def retrieve_relevant_embedding(
    context_files: Optional[Dict[str, str]],
    task_prompt: str,
    *,
    task_category: Optional[str] = None,
    use_mcp: bool = False,  # Новый параметр
    mcp_provider: str = "openai",  # Новый параметр
    mcp_model: Optional[str] = None,  # Новый параметр
    config: Optional[Config] = None,  # Новый параметр
    ...  # остальные параметры
) -> str:
    """
    Retrieval с поддержкой MCP tools
    """
    if use_mcp and task_category:
        from .mcp_retrieval import retrieve_with_mcp
        from pathlib import Path
        
        project_dir = Path(project_dir) if project_dir else None
        if not project_dir:
            logger.warning("project_dir required for MCP retrieval")
            use_mcp = False
        
        if use_mcp:
            try:
                return retrieve_with_mcp(
                    context_files=context_files or {},
                    task_prompt=task_prompt,
                    task_category=task_category,
                    project_dir=project_dir,
                    config=config,
                    provider=mcp_provider,
                    model=mcp_model,
                    use_llm=True,
                )
            except Exception as e:
                logger.warning(f"MCP retrieval failed: {e}. Falling back to standard retrieval.")
    
    # Стандартный retrieval (существующий код)
    ...
```

### Использование в evaluator

```python
# В locobench/evaluation/evaluator.py

result = retrieve_relevant_embedding(
    context_files=context_files,
    task_prompt=task.description,
    project_dir=project_dir,
    task_category=task.category.value,
    use_mcp=True,  # Включить MCP
    mcp_provider="openai",
    mcp_model="gpt-4o",
    config=self.config,
    # ... остальные параметры
)
```

## Конфигурация

### Настройка API ключей

В `config.yaml` или через переменные окружения:

```yaml
api:
  openai_api_key: "sk-..."
  openai_base_url: null  # или кастомный URL
  claude_bearer_token: "sk-ant-..."  # для Anthropic
  default_model_openai: "gpt-4o"
  default_model_claude: "claude-sonnet-4"
```

Или через переменные окружения:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."  # или CLAUDE_BEARER_TOKEN
```

## Поддерживаемые модели

### OpenAI
- `gpt-4o` - Рекомендуется для tool calling
- `gpt-4o-mini` - Более быстрая и дешевая альтернатива
- `o3`, `o3-pro` - Reasoning модели (поддерживают tool calling)
- `gpt-4-turbo` - Legacy модель

### Anthropic
- `claude-sonnet-4` - Рекомендуется (баланс качества и скорости)
- `claude-opus-4` - Максимальное качество
- `claude-sonnet-3.7` - Альтернатива

## Параметры

### `retrieve_with_mcp`

- `context_files`: Dict[str, str] - Доступные файлы проекта
- `task_prompt`: str - Описание задачи
- `task_category`: str - Категория задачи (security_analysis, architectural_understanding, etc.)
- `project_dir`: Path - Директория проекта
- `config`: Optional[Config] - Конфигурация (для API ключей)
- `provider`: str - "openai" или "anthropic"
- `model`: Optional[str] - Имя модели (использует default из config если None)
- `use_llm`: bool - Использовать LLM для tool calling (True) или эвристики (False)

### `MCPLLMIntegrator.retrieve_with_openai/anthropic`

- `model`: str - Имя модели
- `max_iterations`: int - Максимальное количество итераций tool calling (default: 5)

## Как это работает

### 1. Инициализация MCP Server

MCP Server создает набор специализированных tools на основе категории задачи:

```python
mcp_server = LoCoBenchMCPServer(
    task_category="security_analysis",
    ...
)
# Создает tools: find_security_sensitive_files, analyze_dependency_graph_for_security, etc.
```

### 2. LLM получает tools

Tools конвертируются в формат, понятный LLM:

**OpenAI формат:**
```json
{
  "type": "function",
  "function": {
    "name": "find_security_sensitive_files",
    "description": "...",
    "parameters": {...}
  }
}
```

**Anthropic формат:**
```json
{
  "name": "find_security_sensitive_files",
  "description": "...",
  "input_schema": {...}
}
```

### 3. LLM вызывает tools

LLM анализирует задачу и решает, какие tools вызвать:

```python
# LLM видит задачу: "Найти уязвимости в обработке ввода"
# LLM вызывает: find_security_sensitive_files(keywords="input, validation")
# LLM вызывает: find_input_validation_points(input_sources="API, forms")
```

### 4. Tools выполняются

Каждый tool выполняется и возвращает релевантные файлы:

```python
results = tool.execute(keywords="input, validation")
# Возвращает: [{"path": "src/auth.py", "relevance_score": 0.9, ...}, ...]
```

### 5. LLM получает результаты

LLM видит результаты и может запросить дополнительные файлы или завершить:

```python
# LLM видит результаты и решает запросить еще файлы
# LLM вызывает: analyze_dependency_graph_for_security(...)
# Или завершает, если достаточно файлов найдено
```

### 6. Формирование контекста

Финальный контекст формируется из всех выбранных файлов:

```python
context = mcp_server.format_selected_context()
# Возвращает форматированную строку с кодом файлов
```

## Отладка

### Включение логирования

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("locobench.mcp_llm_integration")
```

### Проверка доступных tools

```python
mcp_server = LoCoBenchMCPServer(...)
tools = mcp_server.get_tools_for_llm()
print(f"Available tools: {[t['name'] for t in tools]}")
```

### Тестирование без LLM

```python
# Использовать эвристики вместо LLM
result = retrieve_with_mcp(
    ...,
    use_llm=False,  # Отключить LLM, использовать эвристики
)
```

## Обработка ошибок

Система автоматически fallback на эвристики при ошибках:

```python
try:
    result = retrieve_with_mcp(..., use_llm=True)
except Exception as e:
    # Автоматически fallback на эвристики
    logger.warning(f"MCP failed: {e}. Using heuristics.")
```

## Производительность

- **OpenAI**: ~2-5 секунд на итерацию tool calling
- **Anthropic**: ~3-7 секунд на итерацию tool calling
- **Максимум итераций**: 5 (настраивается через `max_iterations`)

Для ускорения можно:
- Уменьшить `max_iterations` до 2-3
- Использовать более быстрые модели (gpt-4o-mini, claude-sonnet-3.7)
- Использовать `use_llm=False` для эвристик (быстрее, но менее точно)

## Следующие шаги

1. Протестировать на существующих сценариях
2. Сравнить результаты с текущим подходом
3. Оптимизировать prompts для tools
4. Настроить для достижения скора 2.3+
