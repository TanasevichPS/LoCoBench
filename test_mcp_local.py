#!/usr/bin/env python3
"""
Тестовый скрипт для проверки локальных моделей с MCP
"""

import asyncio
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
}


def test_ollama():
    """Тест с Ollama"""
    print("=" * 60)
    print("Тест: MCP с Ollama")
    print("=" * 60)
    
    try:
        result = retrieve_with_mcp(
            context_files=SAMPLE_CONTEXT_FILES,
            task_prompt="Найти уязвимости в обработке пользовательского ввода",
            task_category="security_analysis",
            project_dir=Path("."),
            provider="ollama",
            model="llama3.2",
            base_url="http://localhost:11434",
            use_llm=True,
        )
        
        print(f"\n✅ Результат:\n{result}\n")
        return result
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("💡 Убедитесь, что Ollama запущен: ollama serve")
        print("💡 И модель загружена: ollama pull llama3.2")
        return None


def test_huggingface():
    """Тест с Hugging Face"""
    print("=" * 60)
    print("Тест: MCP с Hugging Face")
    print("=" * 60)
    
    try:
        result = retrieve_with_mcp(
            context_files=SAMPLE_CONTEXT_FILES,
            task_prompt="Найти уязвимости в обработке пользовательского ввода",
            task_category="security_analysis",
            project_dir=Path("."),
            provider="huggingface",
            model="meta-llama/Llama-3.2-3B-Instruct",
            use_llm=True,
        )
        
        print(f"\n✅ Результат:\n{result}\n")
        return result
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("💡 Убедитесь, что установлены зависимости: pip install transformers torch")
        print("💡 И у вас достаточно памяти для модели")
        return None


def test_local_openai():
    """Тест с LocalAI/LM Studio"""
    print("=" * 60)
    print("Тест: MCP с LocalAI/LM Studio")
    print("=" * 60)
    
    try:
        result = retrieve_with_mcp(
            context_files=SAMPLE_CONTEXT_FILES,
            task_prompt="Найти уязвимости в обработке пользовательского ввода",
            task_category="security_analysis",
            project_dir=Path("."),
            provider="local_openai",
            model="llama-3.2",
            base_url="http://localhost:1234",  # LM Studio default
            use_llm=True,
        )
        
        print(f"\n✅ Результат:\n{result}\n")
        return result
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("💡 Убедитесь, что LM Studio запущен и сервер активен")
        print("💡 Или используйте LocalAI на http://localhost:1234")
        return None


def main():
    """Главная функция"""
    print("\n" + "=" * 60)
    print("Тестирование локальных моделей с MCP")
    print("=" * 60 + "\n")
    
    print("Выберите провайдер:")
    print("1. Ollama (рекомендуется)")
    print("2. Hugging Face")
    print("3. LocalAI/LM Studio")
    print("4. Все")
    
    choice = input("\nВаш выбор (1-4): ").strip()
    
    if choice == "1":
        test_ollama()
    elif choice == "2":
        test_huggingface()
    elif choice == "3":
        test_local_openai()
    elif choice == "4":
        test_ollama()
        print()
        test_huggingface()
        print()
        test_local_openai()
    else:
        print("Неверный выбор. Запускаю тест Ollama по умолчанию...")
        test_ollama()
    
    print("=" * 60)
    print("Тестирование завершено")
    print("=" * 60)


if __name__ == "__main__":
    main()
