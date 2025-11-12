# План интеграции MCP Tools для улучшения Retrieval до 2.3+

## Что такое MCP (Model Context Protocol)?

**MCP (Model Context Protocol)** — это протокол от Anthropic, который позволяет LLM взаимодействовать с внешними инструментами и ресурсами через стандартизированный интерфейс.

### Ключевые концепции:

1. **MCP Server** — сервер, который предоставляет инструменты и ресурсы
2. **MCP Tools** — функции, которые LLM может вызывать для выполнения действий
3. **MCP Resources** — данные/файлы, к которым LLM может получить доступ

### Преимущества MCP для Retrieval:

- **Интеллектуальный выбор файлов**: LLM сам решает, какие файлы нужны для задачи
- **Адаптивность**: Разные стратегии для разных типов задач
- **Контекстное понимание**: LLM анализирует задачу и выбирает оптимальный набор файлов
- **Динамическая адаптация**: Может запрашивать дополнительные файлы по мере необходимости

## Текущая ситуация

Максимальный достигнутый скор: **2.115** (в коммите f1922da)
Текущий скор: **~2.0-2.1**
Целевой скор: **2.3+**

### Проблемы текущего подхода:

1. **Статическая конфигурация**: Параметры задаются заранее, не адаптируются к конкретной задаче
2. **Ограниченная семантика**: Embeddings не всегда улавливают сложные связи между файлами
3. **Нет понимания контекста задачи**: Система не понимает, что именно нужно для решения задачи

## Архитектура решения через MCP Tools

### 1. Создание MCP Tools для разных типов задач

Каждый тип задачи будет иметь свой набор специализированных MCP tools:

#### **Security Analysis Tools**
```python
tools = [
    {
        "name": "find_security_sensitive_files",
        "description": "Находит файлы, связанные с безопасностью: аутентификация, авторизация, валидация, шифрование",
        "parameters": {
            "task_prompt": "Описание задачи безопасности",
            "project_structure": "Структура проекта"
        }
    },
    {
        "name": "analyze_dependency_graph_for_security",
        "description": "Анализирует граф зависимостей для поиска потенциальных уязвимостей",
        "parameters": {
            "entry_points": "Точки входа в систему",
            "sensitive_operations": "Операции, требующие проверки безопасности"
        }
    },
    {
        "name": "find_input_validation_points",
        "description": "Находит места, где происходит валидация пользовательского ввода",
        "parameters": {
            "api_endpoints": "API endpoints",
            "user_input_handlers": "Обработчики пользовательского ввода"
        }
    }
]
```

#### **Architectural Understanding Tools**
```python
tools = [
    {
        "name": "identify_core_components",
        "description": "Определяет основные компоненты архитектуры: интерфейсы, абстракции, паттерны",
        "parameters": {
            "task_prompt": "Задача на понимание архитектуры",
            "project_structure": "Структура проекта"
        }
    },
    {
        "name": "map_dependency_hierarchy",
        "description": "Строит иерархию зависимостей между компонентами",
        "parameters": {
            "root_components": "Корневые компоненты",
            "max_depth": "Максимальная глубина анализа"
        }
    },
    {
        "name": "find_design_patterns",
        "description": "Находит паттерны проектирования в коде",
        "parameters": {
            "pattern_types": "Типы паттернов для поиска"
        }
    }
]
```

#### **Code Comprehension Tools**
```python
tools = [
    {
        "name": "trace_execution_flow",
        "description": "Отслеживает поток выполнения кода от точки входа",
        "parameters": {
            "entry_point": "Точка входа",
            "target_function": "Целевая функция для понимания"
        }
    },
    {
        "name": "find_related_functions",
        "description": "Находит функции, связанные с целевой функцией через вызовы",
        "parameters": {
            "function_name": "Имя функции",
            "call_graph": "Граф вызовов"
        }
    },
    {
        "name": "analyze_data_flow",
        "description": "Анализирует поток данных через систему",
        "parameters": {
            "data_sources": "Источники данных",
            "data_sinks": "Приемники данных"
        }
    }
]
```

#### **Feature Implementation Tools**
```python
tools = [
    {
        "name": "find_implementation_examples",
        "description": "Находит похожие реализации для использования как пример",
        "parameters": {
            "feature_type": "Тип функции",
            "similar_features": "Похожие функции"
        }
    },
    {
        "name": "identify_integration_points",
        "description": "Определяет точки интеграции для новой функции",
        "parameters": {
            "feature_requirements": "Требования к функции",
            "existing_apis": "Существующие API"
        }
    },
    {
        "name": "find_related_configurations",
        "description": "Находит конфигурационные файлы, связанные с функцией",
        "parameters": {
            "feature_domain": "Домен функции"
        }
    }
]
```

#### **Bug Investigation Tools**
```python
tools = [
    {
        "name": "trace_error_path",
        "description": "Отслеживает путь ошибки от места возникновения до точки входа",
        "parameters": {
            "error_message": "Сообщение об ошибке",
            "error_location": "Место возникновения ошибки"
        }
    },
    {
        "name": "find_error_handlers",
        "description": "Находит обработчики ошибок, связанные с проблемой",
        "parameters": {
            "error_type": "Тип ошибки",
            "error_context": "Контекст ошибки"
        }
    },
    {
        "name": "analyze_test_coverage",
        "description": "Анализирует покрытие тестами проблемной области",
        "parameters": {
            "problem_area": "Проблемная область",
            "test_files": "Тестовые файлы"
        }
    }
]
```

#### **Cross-File Refactoring Tools**
```python
tools = [
    {
        "name": "identify_refactoring_targets",
        "description": "Определяет файлы, которые нужно рефакторить",
        "parameters": {
            "refactoring_goal": "Цель рефакторинга",
            "code_smells": "Запахи кода"
        }
    },
    {
        "name": "map_cross_file_dependencies",
        "description": "Строит карту зависимостей между файлами для рефакторинга",
        "parameters": {
            "target_files": "Целевые файлы",
            "dependency_types": "Типы зависимостей"
        }
    },
    {
        "name": "find_consolidation_opportunities",
        "description": "Находит возможности для консолидации кода",
        "parameters": {
            "duplicate_patterns": "Дублирующиеся паттерны",
            "similar_code": "Похожий код"
        }
    }
]
```

### 2. Процесс работы с MCP Tools

```
1. Получение задачи
   ↓
2. Определение типа задачи (Security, Architectural, etc.)
   ↓
3. Вызов соответствующего набора MCP tools
   ↓
4. LLM анализирует задачу и вызывает нужные tools
   ↓
5. Tools возвращают релевантные файлы
   ↓
6. LLM может запросить дополнительные файлы через tools
   ↓
7. Формирование финального контекста из выбранных файлов
```

### 3. Реализация MCP Server для LoCoBench

#### Структура MCP Server:

```python
# locobench/mcp_server.py

from typing import Dict, List, Any, Optional
from pathlib import Path
import json

class LoCoBenchMCPServer:
    """MCP Server для интеллектуального retrieval файлов"""
    
    def __init__(self, project_dir: Path, task_category: str):
        self.project_dir = project_dir
        self.task_category = task_category
        self.available_tools = self._get_tools_for_category(task_category)
    
    def _get_tools_for_category(self, category: str) -> List[Dict]:
        """Возвращает набор tools для конкретной категории"""
        category_tools = {
            'security_analysis': self._get_security_tools(),
            'architectural_understanding': self._get_architectural_tools(),
            'code_comprehension': self._get_comprehension_tools(),
            'feature_implementation': self._get_implementation_tools(),
            'bug_investigation': self._get_bug_investigation_tools(),
            'cross_file_refactoring': self._get_refactoring_tools(),
        }
        return category_tools.get(category, [])
    
    def execute_tool(self, tool_name: str, parameters: Dict) -> List[Dict[str, Any]]:
        """Выполняет MCP tool и возвращает релевантные файлы"""
        # Реализация вызова конкретного tool
        pass
    
    def retrieve_with_mcp(self, task_prompt: str, llm_client) -> str:
        """Основной метод retrieval через MCP"""
        # 1. Предоставить LLM доступ к tools
        # 2. LLM вызывает нужные tools
        # 3. Собираем результаты
        # 4. Формируем контекст
        pass
```

### 4. Интеграция с текущей системой

#### Модификация `retrieve_relevant_embedding`:

```python
def retrieve_relevant_embedding(
    ...,
    task_category: Optional[str] = None,
    use_mcp: bool = True,  # Новый параметр
) -> str:
    """
    Retrieval с поддержкой MCP tools для интеллектуального выбора файлов
    """
    if use_mcp and task_category:
        # Использовать MCP-based retrieval
        mcp_server = LoCoBenchMCPServer(project_dir, task_category)
        return mcp_server.retrieve_with_mcp(task_prompt, llm_client)
    else:
        # Использовать текущий подход (fallback)
        return _retrieve_with_current_method(...)
```

## Преимущества подхода через MCP

1. **Адаптивность**: LLM сам выбирает стратегию для каждой задачи
2. **Контекстное понимание**: LLM понимает задачу и выбирает релевантные файлы
3. **Динамичность**: Может запрашивать дополнительные файлы по мере необходимости
4. **Специализация**: Разные tools для разных типов задач
5. **Улучшение качества**: Более точный выбор файлов должен повысить скор

## План реализации

### Этап 1: Базовая инфраструктура MCP
- [ ] Создать базовый класс `LoCoBenchMCPServer`
- [ ] Реализовать систему регистрации tools
- [ ] Интегрировать с LLM клиентом (Anthropic/OpenAI)

### Этап 2: Реализация tools для каждой категории
- [ ] Security Analysis tools
- [ ] Architectural Understanding tools
- [ ] Code Comprehension tools
- [ ] Feature Implementation tools
- [ ] Bug Investigation tools
- [ ] Cross-File Refactoring tools

### Этап 3: Интеграция с текущей системой
- [ ] Модифицировать `retrieve_relevant_embedding`
- [ ] Добавить параметр `use_mcp`
- [ ] Сохранить текущий метод как fallback

### Этап 4: Тестирование и оптимизация
- [ ] Тестирование на существующих сценариях
- [ ] Сравнение результатов с текущим подходом
- [ ] Оптимизация prompts для tools
- [ ] Тонкая настройка для достижения 2.3+

## Пример использования

```python
# В evaluation/evaluator.py

result = retrieve_relevant_embedding(
    context_files=context_files,
    task_prompt=task.description,
    project_dir=project_dir,
    task_category=task.category.value,
    use_mcp=True,  # Включить MCP-based retrieval
    # ... другие параметры
)
```

## Ожидаемые улучшения

- **Security Analysis**: 2.35 → 2.4+ (более точный поиск уязвимостей)
- **Architectural Understanding**: 2.1 → 2.3+ (лучшее понимание структуры)
- **Code Comprehension**: 2.05 → 2.3+ (более точное отслеживание потока)
- **Feature Implementation**: 2.17 → 2.3+ (лучший поиск примеров и точек интеграции)
- **Bug Investigation**: 2.1 → 2.3+ (более точная трассировка ошибок)
- **Cross-File Refactoring**: 2.0 → 2.3+ (лучшее понимание зависимостей)

## Следующие шаги

1. Начать с реализации базовой инфраструктуры MCP
2. Реализовать tools для одной категории (например, Security Analysis)
3. Протестировать и сравнить результаты
4. Постепенно добавлять tools для остальных категорий
5. Оптимизировать для достижения целевого скора 2.3+
