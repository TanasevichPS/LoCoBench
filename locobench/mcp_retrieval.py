"""
MCP-based retrieval system for LoCoBench.

This module provides intelligent file retrieval using Model Context Protocol (MCP) tools.
LLM can dynamically choose which files are needed for a specific task type.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class TaskCategory(Enum):
    """Task categories supported by MCP retrieval"""
    SECURITY_ANALYSIS = "security_analysis"
    ARCHITECTURAL_UNDERSTANDING = "architectural_understanding"
    CODE_COMPREHENSION = "code_comprehension"
    FEATURE_IMPLEMENTATION = "feature_implementation"
    BUG_INVESTIGATION = "bug_investigation"
    CROSS_FILE_REFACTORING = "cross_file_refactoring"
    INTEGRATION_TESTING = "integration_testing"
    MULTI_SESSION_DEVELOPMENT = "multi_session_development"


class MCPTool:
    """Represents an MCP tool for file retrieval"""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, str],
        handler: callable,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary format for LLM"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param: {"type": "string", "description": desc}
                    for param, desc in self.parameters.items()
                },
                "required": list(self.parameters.keys()),
            },
        }
    
    def execute(self, **kwargs) -> List[Dict[str, Any]]:
        """Execute the tool handler"""
        return self.handler(**kwargs)


class LoCoBenchMCPServer:
    """
    MCP Server for intelligent file retrieval in LoCoBench.
    
    Provides specialized tools for different task categories to help LLM
    intelligently select relevant files for each task type.
    """
    
    def __init__(
        self,
        project_dir: Path,
        task_category: str,
        context_files: Dict[str, str],
        task_prompt: str,
    ):
        self.project_dir = project_dir
        self.task_category = task_category
        self.context_files = context_files
        self.task_prompt = task_prompt
        self.selected_files: Set[str] = set()
        self.tools = self._initialize_tools()
    
    def _initialize_tools(self) -> List[MCPTool]:
        """Initialize tools based on task category"""
        category_tools_map = {
            'security_analysis': self._get_security_tools(),
            'architectural_understanding': self._get_architectural_tools(),
            'code_comprehension': self._get_comprehension_tools(),
            'feature_implementation': self._get_implementation_tools(),
            'bug_investigation': self._get_bug_investigation_tools(),
            'cross_file_refactoring': self._get_refactoring_tools(),
            'integration_testing': self._get_integration_testing_tools(),
            'multi_session_development': self._get_multi_session_tools(),
        }
        
        category_lower = self.task_category.lower()
        for key, tools_func in category_tools_map.items():
            if key in category_lower:
                return tools_func()
        
        # Default tools
        return self._get_default_tools()
    
    # ==================== Security Analysis Tools ====================
    
    def _get_security_tools(self) -> List[MCPTool]:
        """Tools for security analysis tasks"""
        return [
            MCPTool(
                name="find_security_sensitive_files",
                description=(
                    "Находит файлы, связанные с безопасностью: "
                    "аутентификация, авторизация, валидация, шифрование, "
                    "обработка паролей, токенов, сессий"
                ),
                parameters={
                    "keywords": "Ключевые слова для поиска (security, auth, encrypt, validate, etc.)",
                    "file_patterns": "Паттерны имен файлов (auth*.py, security*.py, etc.)",
                },
                handler=self._find_security_sensitive_files,
            ),
            MCPTool(
                name="analyze_dependency_graph_for_security",
                description=(
                    "Анализирует граф зависимостей для поиска потенциальных уязвимостей. "
                    "Находит файлы, которые обрабатывают пользовательский ввод или "
                    "выполняют критические операции"
                ),
                parameters={
                    "entry_points": "Точки входа в систему (API endpoints, handlers)",
                    "sensitive_operations": "Операции, требующие проверки безопасности",
                },
                handler=self._analyze_security_dependencies,
            ),
            MCPTool(
                name="find_input_validation_points",
                description=(
                    "Находит места, где происходит валидация пользовательского ввода. "
                    "Важно для поиска потенциальных уязвимостей типа injection"
                ),
                parameters={
                    "input_sources": "Источники ввода (API, forms, files, etc.)",
                },
                handler=self._find_input_validation_points,
            ),
        ]
    
    def _find_security_sensitive_files(
        self,
        keywords: str = "",
        file_patterns: str = "",
    ) -> List[Dict[str, Any]]:
        """Find files related to security"""
        security_keywords = [
            'security', 'auth', 'encrypt', 'validate', 'sanitize',
            'vulnerability', 'exploit', 'attack', 'password', 'token',
            'permission', 'access', 'authorization', 'authentication',
            'session', 'csrf', 'xss', 'sql injection', 'input validation'
        ]
        
        if keywords:
            security_keywords.extend(keywords.split(','))
        
        found_files = []
        for file_path, content in self.context_files.items():
            content_lower = content.lower()
            file_lower = file_path.lower()
            
            # Check keywords in content or filename
            if any(keyword in content_lower or keyword in file_lower 
                   for keyword in security_keywords):
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.9,  # High relevance for security
                    "reason": "Contains security-related code",
                })
        
        return found_files
    
    def _analyze_security_dependencies(
        self,
        entry_points: str = "",
        sensitive_operations: str = "",
    ) -> List[Dict[str, Any]]:
        """Analyze dependency graph for security concerns"""
        # This would analyze imports and call graphs
        # For now, return files that import security-related modules
        security_modules = ['hashlib', 'secrets', 'cryptography', 'jwt', 'bcrypt']
        
        found_files = []
        for file_path, content in self.context_files.items():
            if any(f"import {mod}" in content or f"from {mod}" in content 
                   for mod in security_modules):
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.85,
                    "reason": "Uses security libraries",
                })
        
        return found_files
    
    def _find_input_validation_points(
        self,
        input_sources: str = "",
    ) -> List[Dict[str, Any]]:
        """Find input validation points"""
        validation_patterns = [
            'validate', 'sanitize', 'clean', 'escape', 'filter',
            'input', 'request', 'form', 'parameter', 'query'
        ]
        
        found_files = []
        for file_path, content in self.context_files.items():
            content_lower = content.lower()
            if any(pattern in content_lower for pattern in validation_patterns):
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.8,
                    "reason": "Contains input validation logic",
                })
        
        return found_files
    
    # ==================== Architectural Understanding Tools ====================
    
    def _get_architectural_tools(self) -> List[MCPTool]:
        """Tools for architectural understanding tasks"""
        return [
            MCPTool(
                name="identify_core_components",
                description=(
                    "Определяет основные компоненты архитектуры: "
                    "интерфейсы, абстракции, паттерны проектирования, "
                    "основные модули и их связи"
                ),
                parameters={
                    "component_types": "Типы компонентов для поиска (interface, abstract, pattern)",
                },
                handler=self._identify_core_components,
            ),
            MCPTool(
                name="map_dependency_hierarchy",
                description=(
                    "Строит иерархию зависимостей между компонентами. "
                    "Находит файлы, которые определяют структуру проекта"
                ),
                parameters={
                    "root_components": "Корневые компоненты для анализа",
                    "max_depth": "Максимальная глубина анализа зависимостей",
                },
                handler=self._map_dependency_hierarchy,
            ),
            MCPTool(
                name="find_design_patterns",
                description=(
                    "Находит паттерны проектирования в коде: "
                    "Factory, Builder, Singleton, Strategy, Observer и др."
                ),
                parameters={
                    "pattern_types": "Типы паттернов для поиска",
                },
                handler=self._find_design_patterns,
            ),
        ]
    
    def _identify_core_components(
        self,
        component_types: str = "",
    ) -> List[Dict[str, Any]]:
        """Identify core architectural components"""
        architectural_keywords = [
            'interface', 'abstract', 'pattern', 'component', 'module',
            'structure', 'design', 'architecture', 'framework', 'blueprint',
            'schema', 'layout', 'hierarchy', 'composition', 'decomposition'
        ]
        
        found_files = []
        for file_path, content in self.context_files.items():
            content_lower = content.lower()
            file_lower = file_path.lower()
            
            # Check for architectural indicators
            if any(keyword in content_lower or keyword in file_lower 
                   for keyword in architectural_keywords):
                # Also check for common patterns
                if ('class' in content and ('interface' in content_lower or 
                    'abstract' in content_lower or 'base' in content_lower)):
                    found_files.append({
                        "path": file_path,
                        "content": content,
                        "relevance_score": 0.9,
                        "reason": "Core architectural component",
                    })
        
        return found_files
    
    def _map_dependency_hierarchy(
        self,
        root_components: str = "",
        max_depth: str = "3",
    ) -> List[Dict[str, Any]]:
        """Map dependency hierarchy"""
        # Find files with many imports (likely core components)
        found_files = []
        for file_path, content in self.context_files.items():
            import_count = content.count('import ') + content.count('from ')
            if import_count > 5:  # Files with many dependencies
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.85,
                    "reason": "High dependency count - likely core component",
                })
        
        return found_files
    
    def _find_design_patterns(
        self,
        pattern_types: str = "",
    ) -> List[Dict[str, Any]]:
        """Find design patterns in code"""
        pattern_keywords = [
            'factory', 'builder', 'singleton', 'strategy', 'observer',
            'adapter', 'decorator', 'facade', 'proxy', 'command'
        ]
        
        found_files = []
        for file_path, content in self.context_files.items():
            content_lower = content.lower()
            if any(pattern in content_lower for pattern in pattern_keywords):
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.8,
                    "reason": "Contains design pattern",
                })
        
        return found_files
    
    # ==================== Code Comprehension Tools ====================
    
    def _get_comprehension_tools(self) -> List[MCPTool]:
        """Tools for code comprehension tasks"""
        return [
            MCPTool(
                name="trace_execution_flow",
                description=(
                    "Отслеживает поток выполнения кода от точки входа. "
                    "Находит файлы в порядке вызова функций"
                ),
                parameters={
                    "entry_point": "Точка входа (main function, API endpoint, etc.)",
                    "target_function": "Целевая функция для понимания",
                },
                handler=self._trace_execution_flow,
            ),
            MCPTool(
                name="find_related_functions",
                description=(
                    "Находит функции, связанные с целевой функцией "
                    "через вызовы, импорты или использование"
                ),
                parameters={
                    "function_name": "Имя функции для поиска связей",
                },
                handler=self._find_related_functions,
            ),
            MCPTool(
                name="analyze_data_flow",
                description=(
                    "Анализирует поток данных через систему. "
                    "Находит файлы, которые обрабатывают данные"
                ),
                parameters={
                    "data_sources": "Источники данных",
                    "data_sinks": "Приемники данных",
                },
                handler=self._analyze_data_flow,
            ),
        ]
    
    def _trace_execution_flow(
        self,
        entry_point: str = "",
        target_function: str = "",
    ) -> List[Dict[str, Any]]:
        """Trace execution flow"""
        # Find files with main functions or entry points
        found_files = []
        for file_path, content in self.context_files.items():
            if 'if __name__' in content or 'def main' in content.lower():
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.9,
                    "reason": "Entry point",
                })
        
        return found_files
    
    def _find_related_functions(
        self,
        function_name: str = "",
    ) -> List[Dict[str, Any]]:
        """Find functions related to target function"""
        found_files = []
        for file_path, content in self.context_files.items():
            if function_name and function_name in content:
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.85,
                    "reason": f"References function: {function_name}",
                })
        
        return found_files
    
    def _analyze_data_flow(
        self,
        data_sources: str = "",
        data_sinks: str = "",
    ) -> List[Dict[str, Any]]:
        """Analyze data flow"""
        data_keywords = ['data', 'process', 'transform', 'convert', 'parse']
        
        found_files = []
        for file_path, content in self.context_files.items():
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in data_keywords):
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.8,
                    "reason": "Handles data flow",
                })
        
        return found_files
    
    # ==================== Feature Implementation Tools ====================
    
    def _get_implementation_tools(self) -> List[MCPTool]:
        """Tools for feature implementation tasks"""
        return [
            MCPTool(
                name="find_implementation_examples",
                description=(
                    "Находит похожие реализации для использования как пример. "
                    "Ищет файлы с похожей функциональностью"
                ),
                parameters={
                    "feature_type": "Тип функции для поиска примеров",
                    "similar_features": "Похожие функции в проекте",
                },
                handler=self._find_implementation_examples,
            ),
            MCPTool(
                name="identify_integration_points",
                description=(
                    "Определяет точки интеграции для новой функции: "
                    "API endpoints, service interfaces, configuration files"
                ),
                parameters={
                    "feature_requirements": "Требования к функции",
                },
                handler=self._identify_integration_points,
            ),
            MCPTool(
                name="find_related_configurations",
                description=(
                    "Находит конфигурационные файлы, связанные с функцией: "
                    "settings, config, constants, environment variables"
                ),
                parameters={
                    "feature_domain": "Домен функции",
                },
                handler=self._find_related_configurations,
            ),
        ]
    
    def _find_implementation_examples(
        self,
        feature_type: str = "",
        similar_features: str = "",
    ) -> List[Dict[str, Any]]:
        """Find similar implementation examples"""
        # This would use semantic similarity to find similar code
        # For now, return files with similar structure
        found_files = []
        for file_path, content in self.context_files.items():
            if 'def ' in content and 'class ' in content:
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.8,
                    "reason": "Contains implementation examples",
                })
        
        return found_files
    
    def _identify_integration_points(
        self,
        feature_requirements: str = "",
    ) -> List[Dict[str, Any]]:
        """Identify integration points"""
        integration_keywords = ['api', 'endpoint', 'interface', 'service', 'handler']
        
        found_files = []
        for file_path, content in self.context_files.items():
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in integration_keywords):
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.85,
                    "reason": "Integration point",
                })
        
        return found_files
    
    def _find_related_configurations(
        self,
        feature_domain: str = "",
    ) -> List[Dict[str, Any]]:
        """Find related configuration files"""
        config_patterns = ['config', 'settings', 'constants', 'env', 'yaml', 'json']
        
        found_files = []
        for file_path, content in self.context_files.items():
            file_lower = file_path.lower()
            if any(pattern in file_lower for pattern in config_patterns):
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.8,
                    "reason": "Configuration file",
                })
        
        return found_files
    
    # ==================== Bug Investigation Tools ====================
    
    def _get_bug_investigation_tools(self) -> List[MCPTool]:
        """Tools for bug investigation tasks"""
        return [
            MCPTool(
                name="trace_error_path",
                description=(
                    "Отслеживает путь ошибки от места возникновения до точки входа. "
                    "Находит файлы в стеке вызовов"
                ),
                parameters={
                    "error_message": "Сообщение об ошибке",
                    "error_location": "Место возникновения ошибки",
                },
                handler=self._trace_error_path,
            ),
            MCPTool(
                name="find_error_handlers",
                description=(
                    "Находит обработчики ошибок, связанные с проблемой: "
                    "try-except блоки, error handlers, logging"
                ),
                parameters={
                    "error_type": "Тип ошибки",
                },
                handler=self._find_error_handlers,
            ),
            MCPTool(
                name="analyze_test_coverage",
                description=(
                    "Анализирует покрытие тестами проблемной области. "
                    "Находит тестовые файлы, связанные с проблемным кодом"
                ),
                parameters={
                    "problem_area": "Проблемная область",
                },
                handler=self._analyze_test_coverage,
            ),
        ]
    
    def _trace_error_path(
        self,
        error_message: str = "",
        error_location: str = "",
    ) -> List[Dict[str, Any]]:
        """Trace error path"""
        error_keywords = ['error', 'exception', 'fail', 'raise', 'catch']
        
        found_files = []
        for file_path, content in self.context_files.items():
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in error_keywords):
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.85,
                    "reason": "Error handling code",
                })
        
        return found_files
    
    def _find_error_handlers(
        self,
        error_type: str = "",
    ) -> List[Dict[str, Any]]:
        """Find error handlers"""
        handler_patterns = ['try:', 'except', 'catch', 'error_handler', 'on_error']
        
        found_files = []
        for file_path, content in self.context_files.items():
            if any(pattern in content for pattern in handler_patterns):
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.8,
                    "reason": "Error handler",
                })
        
        return found_files
    
    def _analyze_test_coverage(
        self,
        problem_area: str = "",
    ) -> List[Dict[str, Any]]:
        """Analyze test coverage"""
        test_patterns = ['test_', '_test', 'spec', 'specification']
        
        found_files = []
        for file_path, content in self.context_files.items():
            file_lower = file_path.lower()
            if any(pattern in file_lower for pattern in test_patterns):
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.75,
                    "reason": "Test file",
                })
        
        return found_files
    
    # ==================== Refactoring Tools ====================
    
    def _get_refactoring_tools(self) -> List[MCPTool]:
        """Tools for cross-file refactoring tasks"""
        return [
            MCPTool(
                name="identify_refactoring_targets",
                description=(
                    "Определяет файлы, которые нужно рефакторить: "
                    "дублирующийся код, нарушение DRY, code smells"
                ),
                parameters={
                    "refactoring_goal": "Цель рефакторинга",
                },
                handler=self._identify_refactoring_targets,
            ),
            MCPTool(
                name="map_cross_file_dependencies",
                description=(
                    "Строит карту зависимостей между файлами для рефакторинга. "
                    "Находит все файлы, которые нужно изменить вместе"
                ),
                parameters={
                    "target_files": "Целевые файлы для рефакторинга",
                },
                handler=self._map_cross_file_dependencies,
            ),
        ]
    
    def _identify_refactoring_targets(
        self,
        refactoring_goal: str = "",
    ) -> List[Dict[str, Any]]:
        """Identify refactoring targets"""
        # Find files with many similar patterns (potential duplication)
        found_files = []
        for file_path, content in self.context_files.items():
            # Simple heuristic: files with many similar function definitions
            if content.count('def ') > 10:
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.8,
                    "reason": "Potential refactoring target",
                })
        
        return found_files
    
    def _map_cross_file_dependencies(
        self,
        target_files: str = "",
    ) -> List[Dict[str, Any]]:
        """Map cross-file dependencies"""
        # Find files that import many other files (likely refactoring candidates)
        found_files = []
        for file_path, content in self.context_files.items():
            import_count = content.count('import ') + content.count('from ')
            if import_count > 8:
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.85,
                    "reason": "High cross-file dependencies",
                })
        
        return found_files
    
    # ==================== Integration Testing Tools ====================
    
    def _get_integration_testing_tools(self) -> List[MCPTool]:
        """Tools for integration testing tasks"""
        return [
            MCPTool(
                name="find_integration_points",
                description="Находит точки интеграции между компонентами",
                parameters={
                    "components": "Компоненты для интеграции",
                },
                handler=self._find_integration_points,
            ),
        ]
    
    def _find_integration_points(
        self,
        components: str = "",
    ) -> List[Dict[str, Any]]:
        """Find integration points"""
        integration_keywords = ['integration', 'integrate', 'connect', 'bridge', 'adapter']
        
        found_files = []
        for file_path, content in self.context_files.items():
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in integration_keywords):
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.85,
                    "reason": "Integration point",
                })
        
        return found_files
    
    # ==================== Multi-Session Development Tools ====================
    
    def _get_multi_session_tools(self) -> List[MCPTool]:
        """Tools for multi-session development tasks"""
        return [
            MCPTool(
                name="find_state_management",
                description="Находит файлы, управляющие состоянием между сессиями",
                parameters={
                    "state_type": "Тип состояния",
                },
                handler=self._find_state_management,
            ),
        ]
    
    def _find_state_management(
        self,
        state_type: str = "",
    ) -> List[Dict[str, Any]]:
        """Find state management files"""
        state_keywords = ['state', 'session', 'cache', 'store', 'persist', 'memory']
        
        found_files = []
        for file_path, content in self.context_files.items():
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in state_keywords):
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": 0.85,
                    "reason": "State management",
                })
        
        return found_files
    
    # ==================== Default Tools ====================
    
    def _get_default_tools(self) -> List[MCPTool]:
        """Default tools for unknown task categories"""
        return [
            MCPTool(
                name="find_relevant_files",
                description="Находит файлы, релевантные задаче на основе ключевых слов",
                parameters={
                    "keywords": "Ключевые слова из задачи",
                },
                handler=self._find_relevant_files_default,
            ),
        ]
    
    def _find_relevant_files_default(
        self,
        keywords: str = "",
    ) -> List[Dict[str, Any]]:
        """Default file finder"""
        # Extract keywords from task prompt
        task_words = set(self.task_prompt.lower().split())
        
        found_files = []
        for file_path, content in self.context_files.items():
            content_words = set(content.lower().split())
            overlap = len(task_words & content_words)
            
            if overlap > 5:  # At least 5 common words
                found_files.append({
                    "path": file_path,
                    "content": content,
                    "relevance_score": min(overlap / 20, 0.9),
                    "reason": f"Keyword overlap: {overlap} words",
                })
        
        return found_files
    
    # ==================== Main Retrieval Method ====================
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get tools in format suitable for LLM"""
        return [tool.to_dict() for tool in self.tools]
    
    def execute_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a tool call and return results"""
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    results = tool.execute(**parameters)
                    # Track selected files
                    for result in results:
                        self.selected_files.add(result["path"])
                    return results
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    return []
        
        logger.warning(f"Tool {tool_name} not found")
        return []
    
    def format_selected_context(self) -> str:
        """Format selected files into context string"""
        if not self.selected_files:
            return ""
        
        parts = []
        for file_path in sorted(self.selected_files):
            if file_path in self.context_files:
                parts.append(f"### {file_path}\n```\n{self.context_files[file_path]}\n```")
        
        return "\n\n".join(parts)


def retrieve_with_mcp(
    context_files: Dict[str, str],
    task_prompt: str,
    task_category: str,
    project_dir: Path,
    llm_client=None,  # LLM client for tool calling
) -> str:
    """
    Main entry point for MCP-based retrieval.
    
    This function would integrate with an LLM to intelligently select files
    using MCP tools. For now, it's a placeholder that demonstrates the structure.
    
    Args:
        context_files: Available context files
        task_prompt: Task description
        task_category: Category of the task
        project_dir: Project directory
        llm_client: LLM client for tool calling (Anthropic/OpenAI)
    
    Returns:
        Formatted context string with selected files
    """
    server = LoCoBenchMCPServer(
        project_dir=project_dir,
        task_category=task_category,
        context_files=context_files,
        task_prompt=task_prompt,
    )
    
    # TODO: Integrate with LLM for intelligent tool calling
    # For now, execute default tool as example
    if server.tools:
        default_tool = server.tools[0]
        keywords = " ".join(set(task_prompt.lower().split()[:10]))
        results = default_tool.execute(keywords=keywords)
        
        # Format results
        return server.format_selected_context()
    
    return ""
