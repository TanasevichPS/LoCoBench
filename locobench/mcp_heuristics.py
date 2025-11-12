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
    max_context_tokens: Optional[int] = None,
    top_percent: Optional[float] = None,
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
    
    # Если context_files пустой, попробовать загрузить файлы из project_dir
    # или использовать пути из scenario если они есть
    if not context_files:
        if project_dir and project_dir.exists():
            logger.info(f"📁 context_files пустой, загружаю файлы из project_dir: {project_dir}")
            try:
                from ..retrieval import _collect_project_code_files
                
                project_files = _collect_project_code_files(project_dir)
                context_files = {
                    file_info["path"]: file_info["content"]
                    for file_info in project_files
                }
                logger.info(f"✅ Loaded {len(context_files)} files from project directory")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load files from project_dir: {e}")
                context_files = {}
        else:
            logger.warning(f"⚠️ context_files пустой и project_dir недоступен: {project_dir}")
            return ""
    
    if not context_files:
        logger.warning("⚠️ No context files available for MCP retrieval")
        return ""
    
    # Создать MCP сервер
    server = LoCoBenchMCPServer(
        project_dir=project_dir,
        task_category=task_category,
        context_files=context_files,
        task_prompt=task_prompt,
    )
    
    logger.info(f"📋 Created MCP server with {len(server.tools)} tools, {len(context_files)} files available")
    
    # Выполнить все tools с базовыми параметрами
    all_results = []
    
    for tool in server.tools:
        try:
            # Получить параметры tool из его определения
            tool_params_def = tool.parameters  # Dict с описаниями параметров
            tool_params = {}
            
            # Извлечь ключевые слова из задачи
            task_words = set(task_prompt.lower().split())
            keywords = " ".join(sorted(task_words)[:15])  # Первые 15 уникальных слов
            
            # Заполнить параметры на основе определения tool и типа задачи
            for param_name in tool_params_def.keys():
                if param_name == "keywords":
                    # Добавить keywords с расширением для категории
                    base_keywords = keywords
                    if "security" in task_category.lower():
                        base_keywords += " security auth validate sanitize"
                    elif "architectural" in task_category.lower():
                        base_keywords += " architecture design pattern component"
                    elif "comprehension" in task_category.lower():
                        base_keywords += " trace flow execution call"
                    tool_params[param_name] = base_keywords
                
                elif param_name == "file_patterns":
                    if "security" in task_category.lower():
                        tool_params[param_name] = "auth security validate"
                    else:
                        tool_params[param_name] = ""
                
                elif param_name == "component_types":
                    if "architectural" in task_category.lower():
                        tool_params[param_name] = "interface abstract pattern"
                    else:
                        tool_params[param_name] = ""
                
                elif param_name == "feature_type":
                    # Извлечь тип функции из задачи
                    tool_params[param_name] = keywords.split()[0] if keywords else ""
                
                elif param_name == "similar_features":
                    tool_params[param_name] = keywords
                
                elif param_name == "feature_requirements":
                    tool_params[param_name] = task_prompt[:200]  # Первые 200 символов
                
                elif param_name == "feature_domain":
                    # Извлечь домен из первых слов задачи
                    tool_params[param_name] = keywords.split()[0] if keywords else ""
                
                elif param_name == "function_name":
                    # Попытаться найти имя функции в задаче
                    import re
                    func_match = re.search(r'\b(function|def|method)\s+(\w+)', task_prompt, re.IGNORECASE)
                    tool_params[param_name] = func_match.group(2) if func_match else ""
                
                elif param_name == "entry_point":
                    tool_params[param_name] = "main"  # По умолчанию
                
                elif param_name == "target_function":
                    # Попытаться найти целевую функцию
                    import re
                    func_match = re.search(r'\b(function|def|method)\s+(\w+)', task_prompt, re.IGNORECASE)
                    tool_params[param_name] = func_match.group(2) if func_match else ""
                
                elif param_name == "data_sources" or param_name == "data_sinks":
                    tool_params[param_name] = ""
                
                elif param_name == "error_message" or param_name == "error_location":
                    tool_params[param_name] = ""
                
                elif param_name == "error_type":
                    tool_params[param_name] = ""
                
                elif param_name == "problem_area":
                    tool_params[param_name] = keywords
                
                elif param_name == "refactoring_goal":
                    tool_params[param_name] = task_prompt[:200]
                
                elif param_name == "target_files":
                    tool_params[param_name] = ""
                
                elif param_name == "components":
                    tool_params[param_name] = ""
                
                elif param_name == "state_type":
                    tool_params[param_name] = ""
                
                elif param_name == "input_sources":
                    tool_params[param_name] = "API forms files"
                
                elif param_name == "entry_points" or param_name == "sensitive_operations":
                    tool_params[param_name] = ""
                
                else:
                    # Для неизвестных параметров использовать пустую строку или keywords
                    tool_params[param_name] = keywords if "keyword" in param_name.lower() else ""
            
            # Выполнить tool только с параметрами, которые он принимает
            results = tool.execute(**tool_params)
            all_results.extend(results)
            
            logger.debug(f"✅ Tool '{tool.name}': found {len(results)} files")
            
        except TypeError as e:
            # Ошибка несоответствия параметров - попробовать с минимальными параметрами
            logger.debug(f"⚠️ Tool '{tool.name}' parameter mismatch, trying minimal params: {e}")
            try:
                # Попробовать выполнить без параметров или с пустыми значениями
                minimal_params = {param: "" for param in tool.parameters.keys()}
                results = tool.execute(**minimal_params)
                all_results.extend(results)
                logger.debug(f"✅ Tool '{tool.name}': found {len(results)} files (minimal params)")
            except Exception as e2:
                logger.warning(f"⚠️ Tool '{tool.name}' failed even with minimal params: {e2}")
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
    
    # Применить ограничения на количество файлов
    max_files = None
    if top_percent and context_files:
        max_files = max(1, int(len(context_files) * top_percent))
        logger.debug(f"📊 Limiting to top {max_files} files ({top_percent*100:.1f}% of {len(context_files)} files)")
    
    for result in sorted_results:
        path = result.get("path", "")
        if path and path not in seen_paths:
            seen_paths.add(path)
            unique_results.append(result)
            server.selected_files.add(path)
            
            # Ограничить количество файлов если указано
            if max_files and len(unique_results) >= max_files:
                logger.debug(f"📊 Reached file limit: {max_files} files")
                break
    
    logger.info(f"✅ Selected {len(unique_results)} unique files from {len(all_results)} total results")
    
    # Применить ограничение на размер контекста
    # max_context_tokens уже интерпретируется как количество символов (см. _apply_length_budget в retrieval.py)
    if max_context_tokens:
        max_chars = max_context_tokens  # Используем напрямую как символы
        total_chars = 0
        filtered_results = []
        
        for result in unique_results:
            content = result.get("content", "")
            content_length = len(content)
            
            if total_chars + content_length <= max_chars:
                filtered_results.append(result)
                total_chars += content_length
            else:
                # Попробовать добавить частично, если файл не слишком большой
                remaining = max_chars - total_chars
                if remaining > 1000:  # Минимум 1000 символов для частичного файла
                    # Обрезать файл до оставшегося лимита
                    result_copy = result.copy()
                    result_copy["content"] = content[:remaining] + "\n... [truncated]"
                    filtered_results.append(result_copy)
                    total_chars = max_chars
                break
        
        if len(filtered_results) < len(unique_results):
            logger.info(
                f"📊 Trimmed from {len(unique_results)} to {len(filtered_results)} files "
                f"({total_chars:,} chars, limit: {max_chars:,} chars)"
            )
            unique_results = filtered_results
            server.selected_files = {r.get("path", "") for r in filtered_results}
    
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
