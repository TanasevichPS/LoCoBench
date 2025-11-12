# Установка Ollama для MCP

## Установка Ollama

### Linux (Ubuntu/Debian)

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Или вручную:

```bash
# Скачать и установить
curl -L https://ollama.ai/download/ollama-linux-amd64 -o /usr/local/bin/ollama
chmod +x /usr/local/bin/ollama

# Запустить сервис
ollama serve
```

### Проверка установки

```bash
# Проверить версию
ollama --version

# Загрузить модель
ollama pull llama3.2

# Проверить список моделей
ollama list
```

## Альтернативы: Использование без Ollama

Если не хотите устанавливать Ollama, можно использовать другие варианты:

### Вариант 1: Использовать эвристики (без LLM)

```python
# В config.yaml
retrieval:
  use_mcp: true
  mcp_provider: "ollama"  # Но use_llm=False автоматически использует эвристики
```

Или в коде:
```python
result = retrieve_with_mcp(
    ...,
    use_llm=False,  # Использовать эвристики вместо LLM
)
```

### Вариант 2: Использовать Hugging Face (если установлен)

```bash
pip install transformers torch
```

```python
# В config.yaml
retrieval:
  use_mcp: true
  mcp_provider: "huggingface"
  mcp_model: "meta-llama/Llama-3.2-3B-Instruct"
```

### Вариант 3: Использовать OpenAI/Anthropic (если есть API ключи)

```python
# В config.yaml
retrieval:
  use_mcp: true
  mcp_provider: "openai"  # или "anthropic"
  mcp_model: "gpt-4o"  # или "claude-sonnet-4"
```

### Вариант 4: Использовать LocalAI/LM Studio

Если у вас установлен LM Studio или LocalAI:
```python
retrieval:
  use_mcp: true
  mcp_provider: "local_openai"
  mcp_model: "llama-3.2"
  mcp_base_url: "http://localhost:1234"
```

## Рекомендация

Для начала протестируйте с **эвристиками** (без LLM) - это уже должно дать улучшение:

```python
# test_without_llm.py
from pathlib import Path
from locobench.mcp_retrieval import retrieve_with_mcp

result = retrieve_with_mcp(
    context_files={"file.py": "code..."},
    task_prompt="Найти уязвимости",
    task_category="security_analysis",
    project_dir=Path("."),
    use_llm=False,  # Использовать эвристики
)

print(f"Результат: {len(result)} символов")
```

Это работает без установки каких-либо дополнительных зависимостей!
