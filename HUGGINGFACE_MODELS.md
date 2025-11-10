# Использование Hugging Face моделей в LoCoBench

## Описание

LoCoBench теперь поддерживает использование моделей с Hugging Face Hub для генерации кода и оценки. Это позволяет использовать открытые модели, такие как Qwen2.5-Coder, DeepSeek-Coder и другие.

## Установка зависимостей

```bash
pip install transformers torch
```

## Поддерживаемые модели

### Рекомендуемые модели для кода (небольшие):

1. **Qwen2.5-Coder-1.5B-Instruct** - `Qwen/Qwen2.5-Coder-1.5B-Instruct`
   - Маленькая модель (~1.5B параметров)
   - Хорошо работает для простых задач генерации кода

2. **DeepSeek-Coder-1.3B-Instruct** - `deepseek-ai/deepseek-coder-1.3b-instruct`
   - Маленькая модель (~1.3B параметров)
   - Специализирована на генерации кода

3. **Qwen2.5-Coder-7B-Instruct** - `Qwen/Qwen2.5-Coder-7B-Instruct`
   - Более крупная модель (~7B параметров)
   - Лучше качество, но требует больше памяти

### Другие модели:

- `microsoft/CodeBERT-base` - CodeBERT
- `bigcode/starcoder` - StarCoder
- `codellama/CodeLlama-7b-hf` - CodeLlama

## Использование

### 1. Использование в CLI

```bash
# Оценка модели с Hugging Face
locobench evaluate --model Qwen/Qwen2.5-Coder-1.5B-Instruct --task-category feature_implementation

# Или используя короткое имя (если добавлено в mapping)
locobench evaluate --model qwen-coder-1.5b --task-category feature_implementation
```

### 2. Использование в коде

```python
from locobench.generation.synthetic_generator import MultiLLMGenerator
from locobench.core.config import Config

config = Config.from_yaml("config.yaml")
generator = MultiLLMGenerator(config)

# Генерация с Hugging Face моделью
response = await generator.generate_with_huggingface(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Write a Python function to calculate factorial"
)
```

### 3. Использование с retrieval pipeline

```python
from locobench.retrieval import retrieve_relevant
from locobench.generation.synthetic_generator import MultiLLMGenerator

# Настройка retrieval
config.retrieval.enabled = True
config.retrieval.method = "embedding"
config.retrieval.model_name = "all-MiniLM-L6-v2"

# Получение релевантного контекста
retrieved = retrieve_relevant(
    context_files,
    task_prompt,
    top_k=5,
    method="embedding"
)

# Генерация с использованием контекста
generator = MultiLLMGenerator(config)
response = await generator.generate_with_huggingface(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    f"{task_prompt}\n\nRelevant context:\n{retrieved}"
)
```

## Обработка ответов

Парсер ответов (`llm_parsing.py`) автоматически обрабатывает ответы от Hugging Face моделей. Поддерживаются следующие форматы:

1. **JSON в markdown блоках:**
   ```json
   {
       "files": {
           "file.py": "code here"
       }
   }
   ```

2. **JSON без markdown:**
   ```json
   {"files": {"file.py": "code"}}
   ```

3. **Code blocks:**
   ```python
   def function():
       pass
   ```

4. **Текстовые ответы** с извлечением кода

## Примеры использования

### Полный пайплайн с retrieval:

```python
import asyncio
from locobench.core.config import Config
from locobench.evaluation.evaluator import Evaluator

async def main():
    config = Config.from_yaml("config.yaml")
    
    # Включить retrieval
    config.retrieval.enabled = True
    config.retrieval.method = "embedding"
    
    # Создать evaluator
    evaluator = Evaluator(config)
    
    # Оценить модель Hugging Face
    results = await evaluator.evaluate_models(
        model_names=["Qwen/Qwen2.5-Coder-1.5B-Instruct"],
        scenarios=[...],  # Ваши сценарии
        resume=True
    )
    
    print(results)

asyncio.run(main())
```

## Требования к системе

- **RAM**: Минимум 8GB для моделей 1.5B, 16GB+ для моделей 7B+
- **GPU**: Опционально, но рекомендуется для быстрой генерации
- **Диск**: ~2-5GB для каждой модели (зависит от размера)

## Примечания

1. **Первая загрузка модели** может занять время - модель загружается с Hugging Face Hub
2. **Использование GPU** значительно ускоряет генерацию для больших моделей
3. **Память**: Модели автоматически используют CPU, если CUDA недоступна
4. **Парсинг**: Ответы от Hugging Face моделей обрабатываются тем же парсером, что и ответы от других LLM

## Решение проблем

### Модель не загружается:
```bash
# Проверьте доступ к Hugging Face Hub
huggingface-cli login
```

### Out of Memory:
- Используйте меньшую модель (1.5B вместо 7B)
- Уменьшите max_new_tokens в коде
- Используйте CPU вместо GPU

### Парсинг не работает:
- Убедитесь, что модель генерирует JSON или код
- Проверьте формат ответа модели
- Добавьте более явные инструкции в промпт
