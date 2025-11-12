# Быстрый старт без Ollama

## Проблема

Ollama не установлен, но вы можете использовать MCP с эвристиками (без LLM) или установить зависимости.

## Решение 1: Использовать эвристики (рекомендуется для начала)

MCP tools уже работают с эвристиками без LLM. Это должно дать улучшение по сравнению со стандартным retrieval.

### В config.yaml:

```yaml
retrieval:
  enabled: true
  use_mcp: true
  mcp_provider: "ollama"  # Имя не важно при use_llm=False
  # use_llm будет False по умолчанию в коде
```

### В коде:

MCP автоматически использует эвристики, если LLM недоступен или `use_llm=False`.

## Решение 2: Установить минимальные зависимости

Если хотите использовать LLM, установите зависимости:

```bash
# Для OpenAI/Anthropic
pip install openai anthropic

# Для Hugging Face (локально)
pip install transformers torch

# Для Ollama (если хотите установить)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2
```

## Решение 3: Использовать MCP без LLM в evaluator

MCP tools с эвристиками уже интегрированы и будут работать автоматически:

1. **Включите MCP в config.yaml:**
```yaml
retrieval:
  use_mcp: true
```

2. **Запустите evaluation:**
```bash
python -m locobench.cli evaluate \
    --scenarios data/output/scenarios/test_easy_scenario.json \
    --config config.yaml
```

MCP автоматически использует эвристики, если LLM недоступен.

## Как это работает

1. **С эвристиками (use_llm=False):**
   - MCP Server создает tools для типа задачи
   - Tools выполняются с базовыми параметрами (ключевые слова из задачи)
   - Результаты объединяются и дедуплицируются
   - Формируется контекст из выбранных файлов

2. **С LLM (use_llm=True):**
   - LLM анализирует задачу
   - LLM вызывает нужные tools с оптимальными параметрами
   - Более точный выбор файлов

## Рекомендация

**Начните с эвристик:**
- Не требует дополнительных зависимостей
- Уже интегрировано в код
- Должно дать улучшение
- Позже можно добавить LLM для еще лучших результатов

## Проверка работы

MCP с эвристиками будет работать автоматически, если:
1. `use_mcp: true` в config.yaml
2. `task_category` передается в retrieval
3. `project_dir` указан

При ошибках MCP автоматически fallback на стандартный retrieval.

## Следующие шаги

1. Настройте `config.yaml` с `use_mcp: true`
2. Запустите тестовую оценку
3. Сравните результаты
4. При необходимости установите Ollama для еще лучших результатов
