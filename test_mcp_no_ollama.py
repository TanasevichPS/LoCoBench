#!/usr/bin/env python3
"""
Тест MCP без Ollama - использует эвристики или другие провайдеры
"""

from pathlib import Path
from locobench.mcp_retrieval import retrieve_with_mcp

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
}


def test_with_heuristics():
    """Тест с эвристиками (без LLM)"""
    print("=" * 60)
    print("Тест: MCP с эвристиками (без LLM)")
    print("=" * 60)
    
    try:
        result = retrieve_with_mcp(
            context_files=SAMPLE_CONTEXT_FILES,
            task_prompt="Найти уязвимости в обработке пользовательского ввода",
            task_category="security_analysis",
            project_dir=Path("."),
            use_llm=False,  # Использовать эвристики
        )
        
        print(f"\n✅ Результат получен")
        print(f"   Длина: {len(result)} символов")
        if result:
            print(f"\n   Первые 300 символов:\n{result[:300]}...")
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_huggingface():
    """Тест с Hugging Face (если установлен)"""
    print("\n" + "=" * 60)
    print("Тест: MCP с Hugging Face")
    print("=" * 60)
    
    try:
        # Проверить доступность
        import transformers
        import torch
        
        result = retrieve_with_mcp(
            context_files=SAMPLE_CONTEXT_FILES,
            task_prompt="Найти уязвимости в обработке пользовательского ввода",
            task_category="security_analysis",
            project_dir=Path("."),
            provider="huggingface",
            model="meta-llama/Llama-3.2-3B-Instruct",
            use_llm=True,
        )
        
        print(f"\n✅ Результат получен")
        print(f"   Длина: {len(result)} символов")
        return True
    except ImportError:
        print("⚠️  Hugging Face не установлен. Установите: pip install transformers torch")
        return None
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_openai():
    """Тест с OpenAI (если есть API ключ)"""
    print("\n" + "=" * 60)
    print("Тест: MCP с OpenAI")
    print("=" * 60)
    
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY не установлен. Пропускаем тест.")
        return None
    
    try:
        result = retrieve_with_mcp(
            context_files=SAMPLE_CONTEXT_FILES,
            task_prompt="Найти уязвимости в обработке пользовательского ввода",
            task_category="security_analysis",
            project_dir=Path("."),
            provider="openai",
            model="gpt-4o-mini",  # Более дешевая модель для теста
            use_llm=True,
        )
        
        print(f"\n✅ Результат получен")
        print(f"   Длина: {len(result)} символов")
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Главная функция"""
    print("\n" + "=" * 60)
    print("Тестирование MCP без Ollama")
    print("=" * 60 + "\n")
    
    # Тест 1: Эвристики (всегда работает)
    test_with_heuristics()
    
    # Тест 2: Hugging Face (если установлен)
    test_with_huggingface()
    
    # Тест 3: OpenAI (если есть API ключ)
    test_with_openai()
    
    print("\n" + "=" * 60)
    print("Рекомендация:")
    print("=" * 60)
    print("1. Для начала используйте эвристики (use_llm=False)")
    print("2. Это уже должно дать улучшение по сравнению со стандартным retrieval")
    print("3. Позже можно установить Ollama для еще лучших результатов")
    print("=" * 60)


if __name__ == "__main__":
    main()
