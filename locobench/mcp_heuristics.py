"""
MCP Heuristics-based Retrieval (без LLM)

Простая реализация MCP retrieval с использованием эвристик вместо LLM.
Не требует дополнительных зависимостей (OpenAI, Anthropic, Ollama).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .mcp_retrieval import LoCoBenchMCPServer

logger = logging.getLogger(__name__)


def retrieve_with_mcp_heuristics(
    context_files: Dict[str, str],
    task_prompt: str,
    task_category: str,
    project_dir: Path,
) -> str:
    """
    MCP-based retrieval с использованием эвристик (без LLM).
    
    Это упрощенная версия, которая:
    1. Создает MCP Server с tools для типа задачи
    2. Выполняет все tools с базовыми параметрами
    3. Объединяет результаты и дедуплицирует
    4. Формирует контекст из выбранных файлов
    
    Args:
        context_files: Доступные файлы проекта
        task_prompt: Описание задачи
        task_category: Категория задачи
        project_dir: Директория проекта
    
    Returns:
        Форматированная строка с выбранными файлами
    """
    logger.info(f"🔧 Using MCP heuristics-based retrieval for category: {task_category}")
    
    # Создать MCP сервер
    server = LoCoBenchMCPServer(
        project_dir=project_dir,
        task_category=task_category,
        context_files=context_files,
        task_prompt=task_prompt,
    )
    
    logger.info(f"📋 Created MCP server with {len(server.tools)} tools")
    
    # Выполнить все tools с базовыми параметрами
    all_results = []
    
    for tool in server.tools:
        try:
            # Извлечь ключевые слова из задачи
            task_words = set(task_prompt.lower().split())
            keywords = " ".join(sorted(task_words)[:15])  # Первые 15 уникальных слов
            
            # Выполнить tool с базовыми параметрами
            # Каждый tool может принимать keywords как параметр
            tool_params = {"keywords": keywords}
            
            # Для некоторых tools добавить специфичные параметры
            if "security" in task_category.lower():
                tool_params.update({
                    "keywords": keywords + " security auth validate sanitize",
                    "file_patterns": "auth security validate",
                })
            elif "architectural" in task_category.lower():
                tool_params.update({
                    "keywords": keywords + " architecture design pattern component",
                    "component_types": "interface abstract pattern",
                })
            elif "comprehension" in task_category.lower():
                tool_params.update({
                    "keywords": keywords + " trace flow execution call",
                    "function_name": "",  # Будет извлечено из prompt
                })
            
            results = tool.execute(**tool_params)
            all_results.extend(results)
            
            logger.debug(f"✅ Tool '{tool.name}': found {len(results)} files")
            
        except Exception as e:
            logger.warning(f"⚠️ Tool '{tool.name}' failed: {e}")
            # Продолжить с другими tools
    
    if not all_results:
        logger.warning("⚠️ No files found by any tool, returning empty result")
        return ""
    
    # Дедуплицировать по пути файла
    seen_paths: Set[str] = set()
    unique_results: List[Dict[str, Any]] = []
    
    # Сортировать по relevance_score если доступен
    sorted_results = sorted(
        all_results,
        key=lambda x: x.get("relevance_score", 0.0),
        reverse=True
    )
    
    for result in sorted_results:
        path = result.get("path", "")
        if path and path not in seen_paths:
            seen_paths.add(path)
            unique_results.append(result)
            server.selected_files.add(path)
    
    logger.info(f"✅ Selected {len(unique_results)} unique files from {len(all_results)} total results")
    
    # Форматировать результат
    result = server.format_selected_context()
    
    if result:
        logger.info(f"✅ MCP heuristics retrieval returned {len(result)} characters")
    else:
        logger.warning("⚠️ MCP heuristics retrieval returned empty result")
    
    return result


def retrieve_with_mcp_simple(
    context_files: Dict[str, str],
    task_prompt: str,
    task_category: str,
    project_dir: Path,
    max_files: int = 10,
) -> str:
    """
    Упрощенная версия MCP retrieval с ограничением количества файлов.
    
    Args:
        context_files: Доступные файлы проекта
        task_prompt: Описание задачи
        task_category: Категория задачи
        project_dir: Директория проекта
        max_files: Максимальное количество файлов для возврата
    
    Returns:
        Форматированная строка с выбранными файлами
    """
    result = retrieve_with_mcp_heuristics(
        context_files=context_files,
        task_prompt=task_prompt,
        task_category=task_category,
        project_dir=project_dir,
    )
    
    # Если результат слишком большой, ограничить количество файлов
    if result and max_files > 0:
        lines = result.split("\n")
        file_headers = []
        current_file = []
        files = []
        
        for line in lines:
            if line.startswith("### "):
                if current_file:
                    files.append("\n".join(current_file))
                current_file = [line]
                file_headers.append(line)
            else:
                current_file.append(line)
        
        if current_file:
            files.append("\n".join(current_file))
        
        if len(files) > max_files:
            logger.info(f"📊 Limiting results from {len(files)} to {max_files} files")
            result = "\n\n".join(files[:max_files])
    
    return result
