# Руководство по использованию локальных моделей с MCP

## Обзор

MCP tools теперь поддерживают локальные LLM модели через три провайдера:

1. **Ollama** - Самый простой вариант, поддерживает tool calling
2. **Hugging Face Transformers** - Прямая инференция локальных моделей
3. **LocalAI / LM Studio** - OpenAI-совместимые локальные API

## Установка

### Ollama

```bash
# Установка Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Загрузка модели (например, llama3.2)
ollama pull llama3.2

# Или более мощная модель с поддержкой tool calling
ollama pull llama3.3:70b
```

### Hugging Face Transformers

```bash
pip install transformers torch
```

### LocalAI / LM Studio

- **LM Studio**: Скачайте с https://lmstudio.ai
- **LocalAI**: https://localai.io

## Использование

### 1. Ollama (Рекомендуется)

Ollama - самый простой вариант с нативной поддержкой tool calling.

```python
from pathlib import Path
from locobench.mcp_retrieval import retrieve_with_mcp

result = retrieve_with_mcp(
    context_files=context_files,
    task_prompt="Найти уязвимости безопасности",
    task_category="security_analysis",
    project_dir=Path("."),
    provider="ollama",
    model="llama3.2",  # или "llama3.3:70b" для лучшего качества
    base_url="http://localhost:11434",  # По умолчанию
    use_llm=True,
)
```

**Доступные модели Ollama:**
- `llama3.2` - Быстрая и эффективная (рекомендуется)
- `llama3.3:70b` - Высокое качество, требует больше памяти
- `mistral` - Альтернатива Llama
- `qwen2.5` - Хорошая поддержка tool calling

### 2. Hugging Face Transformers

Прямая инференция локальных моделей. Требует больше памяти, но работает полностью офлайн.

```python
result = retrieve_with_mcp(
    context_files=context_files,
    task_prompt="Найти уязвимости безопасности",
    task_category="security_analysis",
    project_dir=Path("."),
    provider="huggingface",  # или "hf"
    model="meta-llama/Llama-3.2-3B-Instruct",  # Или любая другая модель
    use_llm=True,
)
```

**Рекомендуемые модели Hugging Face:**
- `meta-llama/Llama-3.2-3B-Instruct` - Легкая модель (~6GB RAM)
- `meta-llama/Llama-3.2-7B-Instruct` - Средняя модель (~14GB RAM)
- `mistralai/Mistral-7B-Instruct-v0.2` - Альтернатива
- `Qwen/Qwen2.5-7B-Instruct` - Хорошая поддержка tool calling

**Примечание:** Большинство HF моделей не поддерживают нативный tool calling, поэтому используется симуляция через промпты.

### 3. LocalAI / LM Studio

OpenAI-совместимый API локально. Удобно, если уже используете LM Studio.

```python
result = retrieve_with_mcp(
    context_files=context_files,
    task_prompt="Найти уязвимости безопасности",
    task_category="security_analysis",
    project_dir=Path("."),
    provider="local_openai",  # или "local"
    model="llama-3.2",  # Имя модели в LM Studio
    base_url="http://localhost:1234",  # LM Studio по умолчанию
    api_key=None,  # Обычно не требуется
    use_llm=True,
)
```

**Настройка LM Studio:**
1. Запустите LM Studio
2. Загрузите модель (например, Llama 3.2)
3. Запустите локальный сервер (порт 1234 по умолчанию)
4. Используйте `base_url="http://localhost:1234"`

## Сравнение провайдеров

| Провайдер | Tool Calling | Простота | Память | Скорость |
|-----------|--------------|----------|--------|----------|
| Ollama | ✅ Нативный | ⭐⭐⭐ | Средняя | ⭐⭐⭐ |
| Hugging Face | ⚠️ Симуляция | ⭐⭐ | Высокая | ⭐⭐ |
| LocalAI/LM Studio | ✅ Нативный | ⭐⭐⭐ | Средняя | ⭐⭐⭐ |

## Примеры использования

### Полный пример с Ollama

```python
import asyncio
from pathlib import Path
from locobench.mcp_retrieval import retrieve_with_mcp

async def main():
    context_files = {
        "src/auth.py": "...",
        "src/security.py": "...",
    }
    
    result = retrieve_with_mcp(
        context_files=context_files,
        task_prompt="Найти уязвимости в обработке пользовательского ввода",
        task_category="security_analysis",
        project_dir=Path("."),
        provider="ollama",
        model="llama3.2",
        use_llm=True,
    )
    
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Использование в evaluator

```python
# В locobench/evaluation/evaluator.py

result = retrieve_relevant_embedding(
    context_files=context_files,
    task_prompt=task.description,
    project_dir=project_dir,
    task_category=task.category.value,
    use_mcp=True,
    mcp_provider="ollama",  # Использовать локальную модель
    mcp_model="llama3.2",
    # ... остальные параметры
)
```

## Настройка производительности

### Для Ollama

```python
# Использовать более мощную модель для лучшего качества
result = retrieve_with_mcp(
    ...,
    provider="ollama",
    model="llama3.3:70b",  # Больше памяти, но лучше качество
)
```

### Для Hugging Face

```python
# Использовать меньшую модель для экономии памяти
result = retrieve_with_mcp(
    ...,
    provider="huggingface",
    model="meta-llama/Llama-3.2-3B-Instruct",  # ~6GB RAM
)

# Или большую для лучшего качества
result = retrieve_with_mcp(
    ...,
    provider="huggingface",
    model="meta-llama/Llama-3.2-7B-Instruct",  # ~14GB RAM
)
```

## Troubleshooting

### Ollama не запускается

```bash
# Проверить статус
ollama list

# Перезапустить сервис
ollama serve
```

### Hugging Face модель не загружается

```python
# Проверить доступную память
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Использовать CPU если GPU недостаточно памяти
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Использовать CPU
```

### LocalAI/LM Studio не отвечает

```python
# Проверить доступность сервера
import requests
try:
    response = requests.get("http://localhost:1234/v1/models")
    print(f"Server available: {response.status_code}")
except:
    print("Server not available. Start LM Studio server.")
```

## Рекомендации

1. **Для начала**: Используйте Ollama с `llama3.2` - самый простой вариант
2. **Для лучшего качества**: Используйте `llama3.3:70b` в Ollama
3. **Для полной офлайн работы**: Используйте Hugging Face с локальной моделью
4. **Если уже используете LM Studio**: Используйте `local_openai` провайдер

## Ожидаемая производительность

- **Ollama**: ~2-5 секунд на итерацию tool calling
- **Hugging Face**: ~5-15 секунд на итерацию (зависит от модели и железа)
- **LocalAI/LM Studio**: ~2-5 секунд на итерацию (зависит от модели)

## Преимущества локальных моделей

✅ **Приватность** - Все данные остаются локально
✅ **Без лимитов** - Нет ограничений на количество запросов
✅ **Бесплатно** - После установки нет дополнительных затрат
✅ **Быстро** - Нет задержек сети
✅ **Офлайн** - Работает без интернета

## Недостатки

⚠️ **Требует памяти** - Нужно достаточно RAM/VRAM
⚠️ **Медленнее на CPU** - GPU значительно ускоряет работу
⚠️ **Меньше качество** - Локальные модели обычно слабее облачных

## Следующие шаги

1. Установите Ollama и загрузите модель
2. Протестируйте на простом примере
3. Сравните результаты с облачными моделями
4. Выберите оптимальный вариант для ваших задач
