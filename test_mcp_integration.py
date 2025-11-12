#!/usr/bin/env python3
"""
Тестовый скрипт для проверки интеграции MCP с LLM клиентами
"""

import asyncio
import os
from pathlib import Path
from locobench.mcp_retrieval import LoCoBenchMCPServer, retrieve_with_mcp
from locobench.core.config import Config

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


async def test_mcp_without_llm():
    """Тест MCP без LLM (эвристики)"""
    print("=" * 60)
    print("Тест 1: MCP без LLM (эвристики)")
    print("=" * 60)
    
    result = retrieve_with_mcp(
        context_files=SAMPLE_CONTEXT_FILES,
        task_prompt="Найти уязвимости в обработке пользовательского ввода",
        task_category="security_analysis",
        project_dir=Path("."),
        use_llm=False,  # Использовать эвристики
    )
    
    print(f"\nРезультат:\n{result}\n")
    return result


async def test_mcp_with_openai():
    """Тест MCP с OpenAI"""
    print("=" * 60)
    print("Тест 2: MCP с OpenAI")
    print("=" * 60)
    
    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY не установлен. Пропускаем тест.")
        return None
    
    try:
        config = Config.from_yaml("config.yaml") if Path("config.yaml").exists() else None
        
        result = retrieve_with_mcp(
            context_files=SAMPLE_CONTEXT_FILES,
            task_prompt="Найти уязвимости в обработке пользовательского ввода",
            task_category="security_analysis",
            project_dir=Path("."),
            config=config,
            provider="openai",
            model="gpt-4o-mini",  # Используем более дешевую модель для теста
            use_llm=True,
        )
        
        print(f"\nРезультат:\n{result}\n")
        return result
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_mcp_with_anthropic():
    """Тест MCP с Anthropic"""
    print("=" * 60)
    print("Тест 3: MCP с Anthropic")
    print("=" * 60)
    
    # Проверка наличия API ключа
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("CLAUDE_BEARER_TOKEN"):
        print("⚠️  ANTHROPIC_API_KEY или CLAUDE_BEARER_TOKEN не установлен. Пропускаем тест.")
        return None
    
    try:
        config = Config.from_yaml("config.yaml") if Path("config.yaml").exists() else None
        
        result = retrieve_with_mcp(
            context_files=SAMPLE_CONTEXT_FILES,
            task_prompt="Найти уязвимости в обработке пользовательского ввода",
            task_category="security_analysis",
            project_dir=Path("."),
            config=config,
            provider="anthropic",
            model="claude-sonnet-4",
            use_llm=True,
        )
        
        print(f"\nРезультат:\n{result}\n")
        return result
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_mcp_server_initialization():
    """Тест инициализации MCP сервера"""
    print("=" * 60)
    print("Тест 0: Инициализация MCP сервера")
    print("=" * 60)
    
    server = LoCoBenchMCPServer(
        project_dir=Path("."),
        task_category="security_analysis",
        context_files=SAMPLE_CONTEXT_FILES,
        task_prompt="Найти уязвимости безопасности",
    )
    
    print(f"✅ MCP Server создан")
    print(f"   Категория задачи: {server.task_category}")
    print(f"   Доступно tools: {len(server.tools)}")
    
    tools_info = server.get_tools_for_llm()
    print(f"\nДоступные tools:")
    for tool in tools_info:
        print(f"   - {tool['name']}: {tool['description'][:60]}...")
    
    return server


async def main():
    """Главная функция для запуска всех тестов"""
    print("\n" + "=" * 60)
    print("Тестирование интеграции MCP с LLM")
    print("=" * 60 + "\n")
    
    # Тест 0: Инициализация
    test_mcp_server_initialization()
    print()
    
    # Тест 1: Без LLM
    await test_mcp_without_llm()
    print()
    
    # Тест 2: С OpenAI
    await test_mcp_with_openai()
    print()
    
    # Тест 3: С Anthropic
    await test_mcp_with_anthropic()
    print()
    
    print("=" * 60)
    print("Тестирование завершено")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
