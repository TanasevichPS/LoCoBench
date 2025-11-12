# MCP (Model Context Protocol) - Краткое объяснение

## Что такое MCP?

**MCP (Model Context Protocol)** — это протокол от Anthropic, который позволяет языковым моделям (LLM) взаимодействовать с внешними инструментами и ресурсами через стандартизированный интерфейс.

### Простыми словами:

Вместо того, чтобы система сама решала, какие файлы нужны для задачи, **LLM сам выбирает** нужные файлы, вызывая специальные инструменты (tools).

## Как это работает?

### Традиционный подход (текущий):
```
Задача → Статическая конфигурация → Выбор файлов по параметрам → Результат
```
Проблема: Параметры задаются заранее и не адаптируются к конкретной задаче.

### Подход через MCP:
```
Задача → LLM анализирует задачу → LLM вызывает нужные tools → 
Tools возвращают релевантные файлы → LLM может запросить еще → Результат
```
Преимущество: LLM понимает контекст задачи и выбирает оптимальные файлы.

## Адаптация под разные типы задач

### 1. Security Analysis (Анализ безопасности)

**Проблема**: Нужно найти файлы, связанные с безопасностью, но они могут быть разбросаны по проекту.

**Решение через MCP Tools**:
- `find_security_sensitive_files` — находит файлы с аутентификацией, валидацией, шифрованием
- `analyze_dependency_graph_for_security` — анализирует зависимости для поиска уязвимостей
- `find_input_validation_points` — находит места валидации ввода (важно для поиска injection-уязвимостей)

**Пример использования**:
```python
# LLM видит задачу: "Найти уязвимости в обработке пользовательского ввода"
# LLM вызывает: find_security_sensitive_files(keywords="input, validation, sanitize")
# LLM вызывает: find_input_validation_points(input_sources="API, forms")
# Результат: Точный набор файлов, связанных с безопасностью
```

### 2. Architectural Understanding (Понимание архитектуры)

**Проблема**: Нужно понять структуру проекта, найти основные компоненты и их связи.

**Решение через MCP Tools**:
- `identify_core_components` — находит интерфейсы, абстракции, основные модули
- `map_dependency_hierarchy` — строит иерархию зависимостей
- `find_design_patterns` — находит паттерны проектирования

**Пример использования**:
```python
# LLM видит задачу: "Понять архитектуру системы"
# LLM вызывает: identify_core_components(component_types="interface, abstract")
# LLM вызывает: map_dependency_hierarchy(root_components="main, core")
# Результат: Файлы, определяющие архитектуру проекта
```

### 3. Code Comprehension (Понимание кода)

**Проблема**: Нужно отследить поток выполнения кода, понять, как работает функция.

**Решение через MCP Tools**:
- `trace_execution_flow` — отслеживает поток выполнения от точки входа
- `find_related_functions` — находит функции, связанные с целевой
- `analyze_data_flow` — анализирует поток данных

**Пример использования**:
```python
# LLM видит задачу: "Понять, как работает функция processPayment"
# LLM вызывает: find_related_functions(function_name="processPayment")
# LLM вызывает: trace_execution_flow(entry_point="main", target_function="processPayment")
# Результат: Файлы в порядке вызова функций
```

### 4. Feature Implementation (Реализация функции)

**Проблема**: Нужно найти примеры похожего кода и точки интеграции для новой функции.

**Решение через MCP Tools**:
- `find_implementation_examples` — находит похожие реализации
- `identify_integration_points` — определяет точки интеграции (API, сервисы)
- `find_related_configurations` — находит конфигурационные файлы

**Пример использования**:
```python
# LLM видит задачу: "Реализовать новую функцию авторизации"
# LLM вызывает: find_implementation_examples(feature_type="authentication")
# LLM вызывает: identify_integration_points(feature_requirements="OAuth2")
# Результат: Примеры кода + точки интеграции + конфигурации
```

### 5. Bug Investigation (Расследование багов)

**Проблема**: Нужно отследить путь ошибки от места возникновения до точки входа.

**Решение через MCP Tools**:
- `trace_error_path` — отслеживает путь ошибки
- `find_error_handlers` — находит обработчики ошибок
- `analyze_test_coverage` — анализирует покрытие тестами

**Пример использования**:
```python
# LLM видит задачу: "Найти причину ошибки 'NullPointerException'"
# LLM вызывает: trace_error_path(error_message="NullPointerException", error_location="UserService")
# LLM вызывает: find_error_handlers(error_type="NullPointerException")
# Результат: Файлы в стеке вызовов + обработчики ошибок
```

### 6. Cross-File Refactoring (Рефакторинг между файлами)

**Проблема**: Нужно найти все файлы, которые нужно изменить вместе при рефакторинге.

**Решение через MCP Tools**:
- `identify_refactoring_targets` — определяет файлы для рефакторинга
- `map_cross_file_dependencies` — строит карту зависимостей между файлами

**Пример использования**:
```python
# LLM видит задачу: "Рефакторить дублирующийся код в нескольких файлах"
# LLM вызывает: identify_refactoring_targets(refactoring_goal="remove duplication")
# LLM вызывает: map_cross_file_dependencies(target_files="ServiceA, ServiceB")
# Результат: Все файлы, которые нужно изменить вместе
```

## Преимущества подхода через MCP

1. **Адаптивность**: LLM выбирает стратегию для каждой конкретной задачи
2. **Контекстное понимание**: LLM понимает задачу и выбирает релевантные файлы
3. **Динамичность**: Может запрашивать дополнительные файлы по мере необходимости
4. **Специализация**: Разные tools для разных типов задач
5. **Улучшение качества**: Более точный выбор файлов должен повысить скор до 2.3+

## Текущая реализация

Я создал базовую структуру MCP server в файле `locobench/mcp_retrieval.py`:

- ✅ Класс `LoCoBenchMCPServer` — основной сервер
- ✅ Класс `MCPTool` — представление инструмента
- ✅ Tools для всех типов задач (Security, Architectural, Comprehension, etc.)
- ✅ Базовые реализации handlers для каждого tool

## Следующие шаги

1. **Интеграция с LLM**: Подключить Anthropic/OpenAI API для вызова tools
2. **Улучшение handlers**: Реализовать более интеллектуальную логику поиска файлов
3. **Тестирование**: Протестировать на существующих сценариях
4. **Оптимизация**: Настроить для достижения скора 2.3+

## Пример интеграции

```python
# В retrieval.py добавить параметр use_mcp

def retrieve_relevant_embedding(
    ...,
    task_category: Optional[str] = None,
    use_mcp: bool = True,  # Новый параметр
) -> str:
    if use_mcp and task_category:
        from locobench.mcp_retrieval import retrieve_with_mcp
        return retrieve_with_mcp(
            context_files=context_files,
            task_prompt=task_prompt,
            task_category=task_category,
            project_dir=project_dir,
            llm_client=llm_client,
        )
    else:
        # Текущий подход (fallback)
        return _retrieve_with_current_method(...)
```

## Ожидаемый результат

Переход от статической конфигурации к динамическому выбору файлов через MCP tools должен улучшить качество retrieval и повысить скор с текущих ~2.1 до целевых **2.3+**.
