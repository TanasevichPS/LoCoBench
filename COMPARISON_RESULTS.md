# Результаты сравнения DeepSeek с retrieval и без него

## Статус выполнения

❌ **Запуск не выполнен** - отсутствуют зависимости Python

### Проблема
Не установлены необходимые зависимости из `requirements.txt`, в частности:
- `openai` - для работы с OpenAI API
- `transformers` - для работы с Hugging Face моделями
- `torch` - для локального вывода моделей
- Другие зависимости

## Подготовленные файлы

✅ **Тестовые сценарии созданы:**
- `data/output/scenarios/test_easy_scenario.json` - сценарий easy difficulty (для теста БЕЗ retrieval)
- `data/output/scenarios/test_hard_scenario.json` - сценарий hard difficulty (для теста С retrieval)

✅ **Скрипты для запуска:**
- `run_comparison.py` - автоматический скрипт для сравнения
- `compare_retrieval_full.py` - альтернативный скрипт

## Инструкции для запуска

### Шаг 1: Установка зависимостей

```bash
pip install -r requirements.txt
```

Или установка основных зависимостей:
```bash
pip install transformers torch openai google-generativeai sentence-transformers pyyaml rich click
```

### Шаг 2: Запуск сравнения

**Вариант А: Автоматический скрипт**
```bash
python3 run_comparison.py
```

**Вариант Б: Ручной запуск**

1. БЕЗ retrieval:
```bash
locobench evaluate \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --task-category feature_implementation \
  --difficulty easy \
  --output-file evaluation_results/no_retrieval.json \
  --no-resume
```

2. Включить retrieval в `config.yaml`:
```yaml
retrieval:
  enabled: true
```

3. С retrieval:
```bash
locobench evaluate \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --task-category feature_implementation \
  --difficulty hard \
  --output-file evaluation_results/with_retrieval.json \
  --no-resume
```

## Ожидаемые результаты

После успешного запуска будут созданы файлы:

1. **БЕЗ retrieval:**
   - `evaluation_results/deepseek_no_retrieval_*.json`
   - Метрики для easy difficulty сценариев

2. **С retrieval:**
   - `evaluation_results/deepseek_with_retrieval_*.json`
   - Метрики для hard difficulty сценариев с применением retrieval

3. **Сравнение:**
   - `evaluation_results/comparison_summary.json`
   - Сравнительный анализ результатов

## Метрики для сравнения

Сравнение будет включать:

1. **Общий счет (LCBS)** - общая оценка модели
2. **Software Engineering Score** - оценка качества кода
3. **Functional Correctness Score** - оценка функциональности
4. **Code Quality Score** - оценка качества кода
5. **Long-Context Utilization Score** - использование длинного контекста
6. **Parsing Success Rate** - процент успешного парсинга ответов
7. **Average Generation Time** - среднее время генерации

## Особенности модели DeepSeek-Coder-1.3B

- **Размер:** ~1.3B параметров
- **Тип:** Code generation model
- **Память:** ~2-3GB RAM
- **Время загрузки:** ~30-60 секунд (первый запуск)
- **Время генерации:** ~5-15 секунд на запрос (CPU)

## Примечания

1. **Первая загрузка модели** займет время - модель загружается с Hugging Face Hub
2. **Retrieval** применяется только к hard и expert difficulty сценариям
3. **Parsing** автоматически обрабатывает ответы от Hugging Face моделей
4. **Результаты** сохраняются в формате JSON для последующего анализа

## Файлы результатов

После выполнения будут созданы:
- `evaluation_results/deepseek_no_retrieval_*.json`
- `evaluation_results/deepseek_with_retrieval_*.json`
- `evaluation_results/comparison_summary.json`

## Следующие шаги

1. Установить зависимости: `pip install -r requirements.txt`
2. Запустить скрипт: `python3 run_comparison.py`
3. Проанализировать результаты в `evaluation_results/comparison_summary.json`
