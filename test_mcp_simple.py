#!/usr/bin/env python3
"""
Простой тест MCP без зависимостей от всего пакета
"""

import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

# Прямой импорт модуля
from locobench.mcp_retrieval import LoCoBenchMCPServer, retrieve_with_mcp

# Пример контекстных файлов
SAMPLE_CONTEXT_FILES = {
    "src/auth.py": """
def authenticate_user(username, password):
    if username == "admin" and password == "admin123":
        return True
    return False
""",
    "src/security.py": """
import hashlib

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()
""",
}


def test_server_init():
    """Тест инициализации сервера"""
    print("=" * 60)
    print("Тест: Инициализация MCP Server")
    print("=" * 60)
    
    try:
        server = LoCoBenchMCPServer(
            project_dir=Path("."),
            task_category="security_analysis",
            context_files=SAMPLE_CONTEXT_FILES,
            task_prompt="Найти уязвимости",
        )
        
        print(f"✅ Server создан успешно")
        print(f"   Категория: {server.task_category}")
        print(f"   Tools: {len(server.tools)}")
        print(f"   Имена tools: {[t.name for t in server.tools[:3]]}")
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrieve_without_llm():
    """Тест без LLM (эвристики)"""
    print("\n" + "=" * 60)
    print("Тест: Retrieval без LLM (эвристики)")
    print("=" * 60)
    
    try:
        result = retrieve_with_mcp(
            context_files=SAMPLE_CONTEXT_FILES,
            task_prompt="Найти уязвимости в обработке пользовательского ввода",
            task_category="security_analysis",
            project_dir=Path("."),
            use_llm=False,  # Использовать эвристики
        )
        
        print(f"✅ Результат получен")
        print(f"   Длина результата: {len(result)} символов")
        if result:
            print(f"   Первые 200 символов:\n{result[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Простой тест MCP")
    print("=" * 60 + "\n")
    
    test1 = test_server_init()
    test2 = test_retrieve_without_llm()
    
    print("\n" + "=" * 60)
    if test1 and test2:
        print("✅ Все тесты пройдены!")
    else:
        print("❌ Некоторые тесты не прошли")
    print("=" * 60)
