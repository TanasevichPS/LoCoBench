# Инструкция по запуску последовательных оценок

## Быстрый старт

### Вариант 1: Python скрипт (рекомендуется)
```bash
python3 run_evaluation_comparison.py [MODEL_NAME]
```

Пример:
```bash
python3 run_evaluation_comparison.py DeepSeekR1-70B-LRI
```

### Вариант 2: Bash скрипт
```bash
./run_evaluation_comparison.sh [MODEL_NAME]
```

Пример:
```bash
./run_evaluation_comparison.sh DeepSeekR1-70B-LRI
```

## Что делает скрипт

1. **Создает временные конфигурации:**
   - `config_with_retrieval_YYYYMMDD_HHMMSS.yaml` - с включенным ритривером
   - `config_without_retrieval_YYYYMMDD_HHMMSS.yaml` - с отключенным ритривером

2. **Изолирует результаты и чекпоинты:**
   - С ритривером: `data/output_with_retrieval_TIMESTAMP/`
   - Без ритривера: `data/output_without_retrieval_TIMESTAMP/`
   - Результаты сохраняются в `evaluation_results/with_retrieval_TIMESTAMP/` и `evaluation_results/without_retrieval_TIMESTAMP/`

3. **Запускает оценки последовательно:**
   - Сначала с ритривером (с `--no-resume` для чистого старта)
   - Затем без ритривера (также с `--no-resume`)

## Структура результатов

```
evaluation_results/
├── with_retrieval_20250110_120000/
│   └── evaluation_results.json
└── without_retrieval_20250110_120000/
    └── evaluation_results.json

data/
├── output_with_retrieval_20250110_120000/
│   └── intermediate_results/
│       └── evaluation_checkpoint_MODEL.json
└── output_without_retrieval_20250110_120000/
    └── intermediate_results/
        └── evaluation_checkpoint_MODEL.json
```

## Важные замечания

- ✅ Чекпоинты полностью изолированы - они не будут подгружаться между запусками
- ✅ Каждая оценка запускается с `--no-resume` для гарантии чистого старта
- ✅ Временные конфиги можно удалить после завершения
- ✅ Результаты сохраняются с временными метками для удобного сравнения

## Очистка

После завершения можно удалить временные конфиги:
```bash
rm config_with_retrieval_*.yaml config_without_retrieval_*.yaml
```

Или удалить все временные директории:
```bash
rm -rf data/output_with_retrieval_* data/output_without_retrieval_*
```
