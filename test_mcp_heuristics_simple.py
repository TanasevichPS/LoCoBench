#!/usr/bin/env python3
"""
Простой тест MCP с эвристиками (без LLM)
Работает без дополнительных зависимостей
"""

import sys
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Прямой импорт модуля
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Прямой импорт без загрузки всего пакета
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mcp_heuristics",
        Path(__file__).parent / "locobench" / "mcp_heuristics.py"
    )
    mcp_heuristics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mcp_heuristics)
    
    # Также нужен mcp_retrieval для LoCoBenchMCPServer
    spec2 = importlib.util.spec_from_file_location(
        "mcp_retrieval",
        Path(__file__).parent / "locobench" / "mcp_retrieval.py"
    )
    mcp_retrieval = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mcp_retrieval)
    
except Exception as e:
    print(f"❌ Ошибка импорта: {e}")
    print("💡 Попробуйте запустить через: python -m locobench.cli evaluate")
    sys.exit(1)

# Пример контекстных файлов
SAMPLE_CONTEXT_FILES = {
    "src/auth.py": """
def authenticate_user(username, password):
    # Simple authentication without input validation
    if username == "admin" and password == "admin123":
        return True
    return False
""",
    "src/security.py": """
import hashlib

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

def validate_input(user_input):
    # Basic validation
    if len(user_input) > 100:
        return False
    return True
""",
    "src/api.py": """
from flask import Flask, request

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    # No input sanitization!
    return authenticate_user(username, password)
""",
    "src/main.py": """
from src.api import app

if __name__ == '__main__':
    app.run()
""",
    "src/utils.py": """
def helper_function():
    return "helper"
""",
}


def test_mcp_heuristics():
    """Тест MCP с эвристиками"""
    print("=" * 60)
    print("Тест: MCP с эвристиками (без LLM)")
    print("=" * 60)
    
    try:
        result = mcp_heuristics.retrieve_with_mcp_heuristics(
            context_files=SAMPLE_CONTEXT_FILES,
            task_prompt="Найти уязвимости в обработке пользовательского ввода",
            task_category="security_analysis",
            project_dir=Path("."),
        )
        
        print(f"\n✅ Результат получен")
        print(f"   Длина: {len(result)} символов")
        
        if result:
            print(f"\n   Найдено файлов: {result.count('###')}")
            print(f"\n   Первые 500 символов:\n{result[:500]}...")
            
            # Проверить, что найдены релевантные файлы
            if "auth.py" in result or "security.py" in result or "api.py" in result:
                print("\n✅ Релевантные файлы найдены!")
            else:
                print("\n⚠️  Релевантные файлы не найдены")
        else:
            print("\n⚠️  Результат пустой")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_categories():
    """Тест для разных категорий задач"""
    print("\n" + "=" * 60)
    print("Тест: Разные категории задач")
    print("=" * 60)
    
    categories = [
        ("security_analysis", "Найти уязвимости безопасности"),
        ("architectural_understanding", "Понять архитектуру системы"),
        ("code_comprehension", "Понять, как работает функция login"),
    ]
    
    for category, prompt in categories:
        try:
            result = mcp_heuristics.retrieve_with_mcp_heuristics(
                context_files=SAMPLE_CONTEXT_FILES,
                task_prompt=prompt,
                task_category=category,
                project_dir=Path("."),
            )
            
            file_count = result.count("###") if result else 0
            print(f"   {category}: {file_count} файлов, {len(result)} символов")
            
        except Exception as e:
            print(f"   {category}: ❌ Ошибка - {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Тестирование MCP с эвристиками")
    print("=" * 60 + "\n")
    
    success = test_mcp_heuristics()
    test_different_categories()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Тест пройден!")
        print("\n💡 Использование:")
        print("   1. В config.yaml установите: use_mcp: true")
        print("   2. Запустите evaluation как обычно")
        print("   3. MCP автоматически использует эвристики")
    else:
        print("❌ Тест не пройден")
    print("=" * 60)
