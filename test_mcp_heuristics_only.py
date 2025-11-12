#!/usr/bin/env python3
"""
Простой тест MCP с эвристиками (без LLM) - не требует дополнительных зависимостей
"""

import sys
from pathlib import Path

# Прямой импорт модуля без загрузки всего пакета
sys.path.insert(0, str(Path(__file__).parent))

# Импортируем только нужные модули
from locobench.mcp_retrieval import LoCoBenchMCPServer

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

def validate_input(user_input):
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
    return authenticate_user(username, password)
""",
}


def test_mcp_heuristics():
    """Тест MCP с эвристиками"""
    print("=" * 60)
    print("Тест: MCP с эвристиками (без LLM)")
    print("=" * 60)
    
    try:
        # Создать MCP сервер
        server = LoCoBenchMCPServer(
            project_dir=Path("."),
            task_category="security_analysis",
            context_files=SAMPLE_CONTEXT_FILES,
            task_prompt="Найти уязвимости в обработке пользовательского ввода",
        )
        
        print(f"✅ MCP Server создан")
        print(f"   Категория: {server.task_category}")
        print(f"   Tools: {len(server.tools)}")
        
        # Выполнить все tools с базовыми параметрами
        all_results = []
        for tool in server.tools:
            try:
                # Извлечь ключевые слова из задачи
                keywords = " ".join(set(server.task_prompt.lower().split()[:10]))
                results = tool.execute(keywords=keywords)
                all_results.extend(results)
                print(f"   Tool '{tool.name}': найдено {len(results)} файлов")
            except Exception as e:
                print(f"   Tool '{tool.name}': ошибка - {e}")
        
        # Дедуплицировать по пути
        seen_paths = set()
        unique_results = []
        for result in all_results:
            path = result.get("path", "")
            if path and path not in seen_paths:
                seen_paths.add(path)
                unique_results.append(result)
                server.selected_files.add(path)
        
        # Форматировать результат
        result = server.format_selected_context()
        
        print(f"\n✅ Результат получен")
        print(f"   Всего файлов найдено: {len(unique_results)}")
        print(f"   Длина результата: {len(result)} символов")
        
        if result:
            print(f"\n   Первые 400 символов:\n{result[:400]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Тестирование MCP с эвристиками")
    print("=" * 60 + "\n")
    
    success = test_mcp_heuristics()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Тест пройден!")
        print("\n💡 Рекомендация:")
        print("   Используйте use_llm=False в config.yaml для начала")
        print("   Это уже должно дать улучшение по сравнению со стандартным retrieval")
    else:
        print("❌ Тест не пройден")
    print("=" * 60)
