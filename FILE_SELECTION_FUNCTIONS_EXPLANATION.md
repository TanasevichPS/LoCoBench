# Детальное объяснение функций выбора файлов в генераторе сценариев

## Обзор

Эти функции отвечают за **интеллектуальный отбор файлов** из проекта для создания сценариев оценки. Они определяют, какие файлы включить в `scenario['context_files']`, чтобы достичь нужного уровня сложности и покрытия информации.

---

## 1. `_select_files_with_target_difficulty`

### Назначение
**Принудительно выбирает файлы для достижения конкретного уровня сложности** (easy, medium, hard, expert).

### Когда используется
- Когда нужно создать сценарий с **конкретным уровнем сложности**
- Например: "Создай 1000 сценариев уровня expert"

### Алгоритм работы

```python
async def _select_files_with_target_difficulty(
    task_category: TaskCategory,
    project_files: Dict[str, str],
    project_spec: Dict[str, Any],
    target_difficulty: DifficultyLevel,  # Целевой уровень: EASY, MEDIUM, HARD, EXPERT
    max_retries: int = 5
) -> Tuple[Dict[str, str], float, DifficultyLevel]:
```

#### Шаг 1: Определение целевого диапазона покрытия
```python
# Получает диапазон покрытия для целевой сложности из конфига
target_range = config.phase3.coverage_ranges.get(target_difficulty.value.lower())
# Например, для "expert": [0.80, 1.00]
# Для "hard": [0.60, 0.80]
# Для "medium": [0.40, 0.60]
# Для "easy": [0.20, 0.40]

min_target_coverage, max_target_coverage = target_range
optimal_target_coverage = (min_target_coverage + max_target_coverage) / 2  # Середина диапазона
```

**Диапазоны покрытия (из config.py):**
- **Easy**: 20-40% проекта (маленький контекст)
- **Medium**: 40-60% проекта (средний контекст)
- **Hard**: 60-80% проекта (большой контекст)
- **Expert**: 80-100% проекта (почти весь проект)

#### Шаг 2: Прогрессивные стратегии
Пробует разные стратегии от консервативной до агрессивной:

```python
strategies = [
    ("conservative", optimal_target_coverage),           # Цель: середина диапазона
    ("slightly_aggressive", min + 0.75 * (max - min)),  # Цель: 75% диапазона
    ("aggressive", min + 0.9 * (max - min)),            # Цель: 90% диапазона
    ("very_aggressive", max_target_coverage),            # Цель: максимум диапазона
    ("fallback", min_target_coverage)                   # Цель: минимум диапазона
]
```

**Пример для Expert (диапазон [0.80, 1.00]):**
1. **conservative**: цель 0.90 (середина)
2. **slightly_aggressive**: цель 0.95 (75% от 0.80 до 1.00)
3. **aggressive**: цель 0.98 (90% от 0.80 до 1.00)
4. **very_aggressive**: цель 1.00 (максимум)
5. **fallback**: цель 0.80 (минимум)

#### Шаг 3: Выбор файлов через `_adaptive_file_selection`
Для каждой стратегии вызывает `_adaptive_file_selection` с целевым покрытием:

```python
context_files = await self._adaptive_file_selection(
    task_category, 
    project_files, 
    target_coverage,  # Целевое покрытие для этой стратегии
    aggressive=(strategy_name in ["aggressive", "very_aggressive"])
)
```

#### Шаг 4: Проверка достижения цели
```python
information_coverage = self._calculate_information_coverage(context_files, project_files)
achieved_difficulty = self._determine_difficulty_from_coverage(information_coverage)

if achieved_difficulty == target_difficulty:
    return context_files, information_coverage, achieved_difficulty  # ✅ Успех!
```

#### Шаг 5: Fallback
Если все стратегии не достигли цели, возвращается к естественному отбору:
```python
return await self._select_files_with_coverage_retry(...)
```

### Пример работы

**Задача:** Создать сценарий уровня **Expert** для категории `ARCHITECTURAL_UNDERSTANDING`

1. **Целевой диапазон**: [0.80, 1.00] (expert)
2. **Стратегия 1 (conservative)**: цель 0.90
   - Выбирает файлы до достижения 0.90 покрытия
   - Получено: 0.85 покрытия → **HARD** (не подходит)
3. **Стратегия 2 (slightly_aggressive)**: цель 0.95
   - Выбирает больше файлов
   - Получено: 0.92 покрытия → **EXPERT** ✅
4. **Возвращает**: отобранные файлы с покрытием 0.92

---

## 2. `_adaptive_file_selection`

### Назначение
**Интеллектуально выбирает файлы для достижения целевого покрытия информации**, учитывая категорию задачи.

### Когда используется
- Вызывается из `_select_files_with_target_difficulty`
- Вызывается из `_select_files_with_coverage_retry`
- Основная функция для адаптивного выбора файлов

### Алгоритм работы

```python
async def _adaptive_file_selection(
    task_category: TaskCategory,      # Категория задачи
    project_files: Dict[str, str],     # Все файлы проекта
    target_coverage: float,            # Целевое покрытие (0.0 - 1.0)
    aggressive: bool = False           # Агрессивный режим (предпочитает большие файлы)
) -> Dict[str, str]:
```

#### Шаг 1: Выбор базовых файлов для категории
```python
# Получает список "core files" для категории задачи
core_files = self._get_core_files_for_category(task_category, project_files)

# Добавляет core files в выборку
for file_path in core_files:
    selected_files[file_path] = project_files[file_path]
```

**Core files по категориям:**
- **ARCHITECTURAL_UNDERSTANDING**: `main.`, `app.`, `index.`, `config.`, `module.`, `package.json`, `Cargo.toml`
- **FEATURE_IMPLEMENTATION**: `service.`, `api.`, `controller.`, `handler.`, `model.`, `repository.`
- **BUG_INVESTIGATION**: `test.`, `spec.`, `error.`, `exception.`, `log.`, `debug.`
- **SECURITY_ANALYSIS**: `auth.`, `security.`, `login.`, `password.`, `token.`, `crypto.`
- И т.д.

#### Шаг 2: Проверка текущего покрытия
```python
current_coverage = self._calculate_information_coverage(selected_files, project_files)

if current_coverage >= target_coverage:
    return selected_files  # ✅ Уже достигли цели!
```

**Формула покрытия:**
```python
coverage = (сумма размеров выбранных файлов) / (сумма размеров всех файлов проекта)
```

#### Шаг 3: Ранжирование оставшихся файлов
```python
remaining_files = {k: v for k, v in project_files.items() if k not in selected_files}
ranked_files = self._rank_files_by_relevance(task_category, remaining_files, aggressive)
```

**Ранжирование учитывает:**
1. **Релевантность имени файла** (40% веса)
   - Проверяет наличие ключевых слов категории в имени файла
   - Например, для `FEATURE_IMPLEMENTATION`: `feature`, `implement`, `api`, `endpoint`
2. **Размер файла** (30% веса)
   - Большие файлы часто важнее
   - Нормализация: `min(size / 10000, 1.0)`
3. **Сложность файла** (20% веса)
   - Подсчет функций, классов, циклов, обработки ошибок
4. **Агрессивный режим** (+50% к размеру)
   - Если `aggressive=True`, большие файлы получают дополнительный бонус

#### Шаг 4: Добавление файлов до достижения цели
```python
for file_path, score in ranked_files:  # Отсортированы по релевантности
    selected_files[file_path] = project_files[file_path]
    current_coverage = self._calculate_information_coverage(selected_files, project_files)
    
    if current_coverage >= target_coverage:
        break  # ✅ Достигли целевого покрытия!
```

### Пример работы

**Задача:** Выбрать файлы для `FEATURE_IMPLEMENTATION` с покрытием 0.50 (medium)

1. **Core files**: `UserService.java`, `ProductController.java`, `OrderRepository.java`
   - Покрытие: 0.15 (недостаточно)

2. **Ранжирование оставшихся файлов:**
   - `FeatureManager.java` (score: 8.5) - содержит "feature" в имени
   - `ApiClient.java` (score: 7.2) - содержит "api" в имени
   - `Utils.java` (score: 3.1) - не релевантен
   - `TestHelper.java` (score: 2.5) - не релевантен

3. **Добавление файлов:**
   - Добавляет `FeatureManager.java` → покрытие: 0.28
   - Добавляет `ApiClient.java` → покрытие: 0.42
   - Добавляет `Utils.java` → покрытие: 0.52 ✅

4. **Результат**: 5 файлов с покрытием 0.52

---

## 3. `_select_context_files`

### Назначение
**Простой выбор файлов на основе категории задачи**, без учета целевого покрытия. Используется как fallback или для простых случаев.

### Когда используется
- В стратегии `category_focused` в `_select_files_with_coverage_retry`
- Как fallback, когда адаптивный выбор не работает
- Для быстрого выбора файлов без сложных вычислений

### Алгоритм работы

```python
def _select_context_files(
    task_category: TaskCategory,
    project_files: Dict[str, str]
) -> Dict[str, str]:
```

#### Разные стратегии по категориям:

**1. ARCHITECTURAL_UNDERSTANDING:**
```python
# Фокус на основных файлах реализации
return self._select_files_by_pattern(project_files, ['src/', 'main.', 'app.', 'server.'])
```
- Ищет файлы в `src/`, с именами `main.*`, `app.*`, `server.*`

**2. CROSS_FILE_REFACTORING:**
```python
# Включает несколько связанных файлов
return self._select_random_subset(project_files, min_files=3, max_files=8)
```
- Случайно выбирает 3-8 файлов (для рефакторинга нужно несколько файлов)

**3. FEATURE_IMPLEMENTATION:**
```python
# Основные файлы, куда добавляются новые функции
return self._select_files_by_pattern(project_files, ['src/', 'lib/', 'core/'])
```
- Ищет файлы в `src/`, `lib/`, `core/`

**4. BUG_INVESTIGATION:**
```python
# Смесь файлов реализации и тестов
return self._select_files_by_pattern(project_files, ['src/', 'test', 'spec'])
```
- Ищет файлы в `src/` и тестовые файлы

**5. MULTI_SESSION_DEVELOPMENT:**
```python
# Более широкий контекст для многосессионной работы
return self._select_random_subset(project_files, min_files=5, max_files=12)
```
- Случайно выбирает 5-12 файлов (нужен широкий контекст)

**6. CODE_COMPREHENSION:**
```python
# Фокус на сложных файлах реализации
return self._select_files_by_complexity(project_files, target_count=4)
```
- Выбирает 4 самых сложных файла (по размеру)

**7. INTEGRATION_TESTING:**
```python
# Тестовые файлы и точки интеграции
return self._select_files_by_pattern(project_files, ['test', 'spec', 'integration', 'api/'])
```
- Ищет тестовые файлы и API файлы

**8. SECURITY_ANALYSIS:**
```python
# Файлы, связанные с безопасностью
return self._select_files_by_pattern(project_files, ['auth', 'security', 'config', 'env'])
```
- Ищет файлы с ключевыми словами безопасности

**9. Default (если категория не распознана):**
```python
return self._select_random_subset(project_files, min_files=2, max_files=6)
```
- Случайно выбирает 2-6 файлов

### Вспомогательные функции:

**`_select_files_by_pattern`:**
```python
# Ищет файлы, содержащие паттерны в пути
for file_path, content in project_files.items():
    if any(pattern.lower() in file_path.lower() for pattern in patterns):
        selected[file_path] = content
```

**`_select_random_subset`:**
```python
# Случайно выбирает N файлов
file_list = list(project_files.items())
count = min(max_files, max(min_files, len(file_list)))
selected_items = random.sample(file_list, count)
return dict(selected_items)
```

**`_select_files_by_complexity`:**
```python
# Сортирует файлы по размеру и выбирает самые большие
sorted_files = sorted(project_files.items(), key=lambda x: len(x[1]), reverse=True)
return dict(sorted_files[:target_count])
```

### Пример работы

**Задача:** Выбрать файлы для `BUG_INVESTIGATION`

1. **Паттерны**: `['src/', 'test', 'spec']`
2. **Поиск файлов:**
   - `src/main/java/UserService.java` ✅ (содержит `src/`)
   - `src/test/java/UserServiceTest.java` ✅ (содержит `test`)
   - `src/spec/UserServiceSpec.java` ✅ (содержит `spec`)
   - `docs/README.md` ❌ (не подходит)

3. **Результат**: 3 файла

---

## Сравнение функций

| Функция | Цель | Использование | Сложность |
|---------|------|---------------|-----------|
| `_select_files_with_target_difficulty` | Достичь конкретной сложности | Принудительное создание сценариев нужной сложности | Высокая (множество стратегий) |
| `_adaptive_file_selection` | Достичь целевого покрытия | Основная функция адаптивного выбора | Средняя (ранжирование + покрытие) |
| `_select_context_files` | Простой выбор по категории | Fallback или простые случаи | Низкая (паттерны или случайный выбор) |

---

## Взаимосвязь функций

```
_create_scenario()
    │
    ├─> Если target_difficulty задан:
    │       └─> _select_files_with_target_difficulty()
    │               └─> _adaptive_file_selection() [много раз с разными стратегиями]
    │                       ├─> _get_core_files_for_category()
    │                       ├─> _rank_files_by_relevance()
    │                       └─> _calculate_information_coverage()
    │
    └─> Если target_difficulty НЕ задан:
            └─> _select_files_with_coverage_retry()
                    ├─> Стратегия 1: _adaptive_file_selection()
                    ├─> Стратегия 2: _adaptive_file_selection(aggressive=True)
                    ├─> Стратегия 3: _select_context_files() + _expand_file_selection()
                    └─> Стратегия 4: _select_context_files() [fallback]
```

---

## Ключевые метрики

### Information Coverage (Покрытие информации)
```python
coverage = (сумма размеров выбранных файлов) / (сумма размеров всех файлов проекта)
```

**Диапазоны по сложности:**
- Easy: 0.20 - 0.40 (20-40% проекта)
- Medium: 0.40 - 0.60 (40-60% проекта)
- Hard: 0.60 - 0.80 (60-80% проекта)
- Expert: 0.80 - 1.00 (80-100% проекта)

### Relevance Score (Оценка релевантности)
```python
score = filename_relevance * 0.4 + size_score * 0.3 + complexity * 0.2
if aggressive:
    score += size_score * 0.5  # Дополнительный бонус большим файлам
```

---

## Выводы

1. **`_select_files_with_target_difficulty`** - самая сложная функция, использует прогрессивные стратегии для достижения конкретной сложности
2. **`_adaptive_file_selection`** - основная функция, интеллектуально выбирает файлы с учетом категории и целевого покрытия
3. **`_select_context_files`** - простая функция, использует паттерны или случайный выбор для быстрого отбора файлов

Все три функции работают вместе, чтобы создать оптимальный набор файлов для каждого сценария оценки, учитывая категорию задачи и требуемый уровень сложности.
