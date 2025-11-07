# Исправление логики поиска проектов для ритривера

## Проблема

Ритривер не мог найти проекты, потому что неправильно парсил `scenario_id` и искал проекты не в той структуре директорий.

## Структура данных

### Формат scenario_id:
```
{base_id}_{task_category}_{difficulty}_{num}
```

**Примеры:**
- `test_python_easy_001_feature_implementation_easy_01`
- `c_desktop_productivity_expert_091_integration_testing_expert_01`

### Структура директорий:
```
data/generated/
  └── {base_id}/                    # Например: test_python_easy_001
      └── {project_name}/            # Например: my_project или beatboard_studio
          ├── src/
          │   ├── main.c
          │   └── ...
          └── tests/
              └── ...
```

## Что было исправлено

### Старая логика:
- Искала проекты напрямую в `data/generated/` по полному `scenario_id`
- Не учитывала структуру `{base_id}/{project_name}/`

### Новая логика:

1. **Извлечение base_id из scenario_id:**
   - Ищет первое вхождение `_{task_category}_` в `scenario_id`
   - Берет все до этого вхождения как `base_id`
   - Пример: `test_python_easy_001_feature_implementation_easy_01` → `base_id = "test_python_easy_001"`

2. **Поиск проекта:**
   - Ищет `data/generated/{base_id}/`
   - Внутри ищет подпапки с проектами
   - Если одна подпапка → использует её
   - Если несколько → использует первую
   - Если нет подпапок → использует саму `base_id` папку

3. **Fallback логика:**
   - Если `base_id` не найден, пытается найти похожие директории
   - Логирует доступные директории для отладки

## Примеры работы

### Пример 1: Стандартная структура
```
scenario_id: "test_python_easy_001_feature_implementation_easy_01"
base_id: "test_python_easy_001"
Ищет: data/generated/test_python_easy_001/
Находит: data/generated/test_python_easy_001/my_project/
Использует: data/generated/test_python_easy_001/my_project/
```

### Пример 2: Несколько проектов
```
scenario_id: "c_desktop_productivity_expert_091_integration_testing_expert_01"
base_id: "c_desktop_productivity_expert_091"
Ищет: data/generated/c_desktop_productivity_expert_091/
Находит: 
  - data/generated/c_desktop_productivity_expert_091/project1/
  - data/generated/c_desktop_productivity_expert_091/project2/
Использует: project1 (первый найденный)
```

### Пример 3: Файлы напрямую в base_id
```
scenario_id: "test_python_easy_001_feature_implementation_easy_01"
base_id: "test_python_easy_001"
Ищет: data/generated/test_python_easy_001/
Находит: нет подпапок, но есть файлы напрямую
Использует: data/generated/test_python_easy_001/ (саму папку)
```

## Изменения в коде

**Файл:** `locobench/evaluation/evaluator.py`

**Строки:** 2226-2329

**Основные изменения:**
1. Использует `TaskCategory` enum для получения всех возможных task categories
2. Использует `DifficultyLevel` enum для получения всех возможных difficulty levels
3. Правильно извлекает `base_id` из `scenario_id`
4. Ищет проекты в правильной структуре: `{base_id}/{project_name}/`
5. Добавлено подробное логирование для отладки

## Результат

Теперь ритривер:
- ✅ Правильно находит проекты по `scenario_id`
- ✅ Поддерживает структуру `{base_id}/{project_name}/`
- ✅ Имеет fallback логику для разных структур
- ✅ Логирует процесс поиска для отладки
