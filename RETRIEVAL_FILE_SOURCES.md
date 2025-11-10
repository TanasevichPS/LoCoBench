# Откуда ритривер берет файлы? 🔍

## Полный путь данных

### 1. Phase 1-2: Генерация проектов
```
data/generated/
  └── c_desktop_productivity_expert_091/
      └── beatboard_studio/
          ├── src/
          │   ├── main.c
          │   ├── include/
          │   │   └── beatboard_plugin_api.h
          │   ├── core/
          │   │   ├── component_manager.c
          │   │   └── event_bus.c
          │   ├── components/
          │   │   ├── persistence_component.c
          │   │   ├── ui_component.c
          │   │   ├── audio_component.c
          │   │   ├── plugin_component.c
          │   │   └── updater_component.c
          │   └── models/
          │       └── app_state.h
          ├── tests/
          │   └── test_persistence.c
          └── docs/
              └── plugin_development.md
```

### 2. Phase 3: Генерация сценариев

Когда создается сценарий (`scenario_generator.py`, строка 202):
```python
"context_files": list(context_files.keys())
```

**Важно**: В сценарий сохраняются только **пути к файлам**, а не их содержимое!

Пример сценария (`data/output/scenarios/...json`):
```json
{
  "id": "c_desktop_productivity_medium_019_integration_testing_expert_01",
  "context_files": [
    "src/include/beatboard_plugin_api.h",
    "tests/test_persistence.c",
    "src/main.c",
    "src/core/component_manager.c",
    ...
  ],
  "metadata": {
    "project_path": "./data/generated/c_desktop_productivity_expert_091/beatboard_studio"
  }
}
```

### 3. Phase 4: Оценка моделей - Ритривер загружает файлы

Ритривер работает в `evaluator.py` (строки 2206-2343):

#### Шаг 1: Определение директории проекта

```python
# Строка 2224-2225: Проверяет metadata сценария
if 'project_path' in metadata:
    project_dir = Path(metadata['project_path'])
```

Если `project_path` не указан, пытается найти по `scenario_id`:

```python
# Строка 2228: Берет директорию из конфига
generated_dir = Path(self.config.data.generated_dir)  # ./data/generated

# Строка 2256: Ищет подходящую папку проекта
for project_folder in generated_dir.iterdir():
    if project_folder.is_dir():
        # Пытается найти совпадение по имени
        # Например: scenario_id = "c_desktop_productivity_medium_019..."
        # Ищет: "c_desktop_productivity_expert_091"
```

#### Шаг 2: Загрузка файлов из проекта

```python
# Строка 2295-2312: Для каждого файла из context_files
for file_path in context_files_list:
    # file_path = "src/include/beatboard_plugin_api.h"
    
    # Строка 2301: Строит полный путь
    file_full_path = project_dir / file_path
    # = data/generated/c_desktop_productivity_expert_091/beatboard_studio/src/include/beatboard_plugin_api.h
    
    # Строка 2302-2305: Читает файл
    if file_full_path.exists():
        with open(file_full_path, 'r', encoding='utf-8') as f:
            context_files_content[file_path] = f.read()
```

#### Шаг 3: Ритривер выбирает релевантные фрагменты

```python
# Строка 2325-2330: Вызывает ритривер
retrieved_context = retrieve_relevant(
    context_files_content,  # Словарь: путь -> содержимое файла
    task_prompt_text,       # Описание задачи
    top_k=5                 # Выбрать 5 самых релевантных фрагментов
)
```

## Проблема в вашем случае

### Что происходит:

1. ✅ Сценарий создан с `context_files` = список путей
2. ❌ Ритривер пытается найти проект в `data/generated/`
3. ❌ Проект не найден (директория не существует или имя не совпадает)
4. ❌ Файлы не загружены
5. ⚠️ Ритривер работает без контекста

### Почему файлы не найдены:

**Вариант 1**: Директория `data/generated/` не существует
```
⚠️ Generated directory does not exist: ./data/generated
```

**Вариант 2**: Проект существует, но имя не совпадает
```
scenario_id = "c_desktop_productivity_medium_019_integration_testing_expert_01"
Ищет проект: "c_desktop_productivity_expert_091"
Не находит совпадение!
```

**Вариант 3**: Файлы не существуют в проекте
```
Context file not found: data/generated/.../beatboard_studio/src/include/beatboard_plugin_api.h
```

## Решение

### Вариант 1: Убедитесь, что проекты сгенерированы

```bash
# Проверьте, что директория существует
ls -la data/generated/

# Должны быть папки проектов:
# c_desktop_productivity_expert_091/
# python_web_ecommerce_easy_001/
# и т.д.
```

### Вариант 2: Добавьте project_path в metadata сценария

Если проект существует, но не находится автоматически, можно указать путь явно:

```json
{
  "id": "...",
  "context_files": [...],
  "metadata": {
    "project_path": "./data/generated/c_desktop_productivity_expert_091/beatboard_studio"
  }
}
```

### Вариант 3: Сохраняйте содержимое файлов в сценарий

Можно изменить `scenario_generator.py`, чтобы сохранять содержимое файлов:

```python
# Вместо:
"context_files": list(context_files.keys())

# Использовать:
"context_files": context_files  # Словарь: путь -> содержимое
```

Но это увеличит размер файлов сценариев!

## Визуальная схема

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1-2: Генерация проектов                              │
│ data/generated/c_desktop_productivity_expert_091/          │
│   └── beatboard_studio/                                     │
│       ├── src/main.c          ← Файлы проекта               │
│       └── tests/test_persistence.c                          │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Генерация сценариев                                │
│ scenario_generator.py                                       │
│   - Загружает файлы из проекта                             │
│   - Выбирает подмножество для контекста                     │
│   - Сохраняет ТОЛЬКО ПУТИ в сценарий:                       │
│     "context_files": ["src/main.c", ...]                   │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Оценка моделей - Ритривер                          │
│ evaluator.py (строки 2206-2343)                             │
│                                                              │
│ 1. Читает сценарий:                                         │
│    context_files = ["src/main.c", ...]                     │
│                                                              │
│ 2. Ищет проект:                                             │
│    project_dir = data/generated/.../beatboard_studio/     │
│                                                              │
│ 3. Загружает файлы:                                         │
│    for file_path in context_files:                          │
│        full_path = project_dir / file_path                  │
│        content = read_file(full_path)                       │
│                                                              │
│ 4. Ритривер выбирает релевантные фрагменты                 │
└─────────────────────────────────────────────────────────────┘
```

## Итого

**Ритривер берет файлы из:**
1. `data/generated/{project_name}/` - директория проекта
2. Пути к файлам указаны в сценарии в поле `context_files`
3. Полный путь = `project_dir / file_path` из сценария

**Проблема:** Если проект не найден или файлы не существуют, ритривер не может загрузить контекст.
