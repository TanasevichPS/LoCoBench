#!/usr/bin/env python3
"""
Минимальный тест для сравнения retrieval - создает тестовые сценарии и запускает оценку
"""

import json
import sys
from pathlib import Path

# Создать тестовые сценарии
def create_test_scenarios():
    """Создать тестовые сценарии для оценки"""
    scenarios_dir = Path("data/output/scenarios")
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    
    # Тестовый сценарий 1: Easy difficulty
    scenario1 = {
        "id": "test_python_easy_001",
        "title": "Calculate Factorial Function",
        "description": "Implement a factorial function in Python",
        "difficulty": "easy",
        "task_category": "feature_implementation",
        "language": "python",
        "task_prompt": "Write a Python function `factorial(n)` that calculates the factorial of a number n. The function should handle edge cases (n=0, n=1) and use recursion. Return JSON format: {\"files\": {\"factorial.py\": \"...\"}}",
        "context_files": {
            "utils.py": """def calculate_sum(a, b):
    '''Calculate sum of two numbers'''
    return a + b

def calculate_product(a, b):
    '''Calculate product of two numbers'''
    return a * b"""
        },
        "metadata": {
            "project_path": "./data/generated/test_project",
            "context_length": 500
        }
    }
    
    # Тестовый сценарий 2: Hard difficulty (для проверки retrieval)
    scenario2 = {
        "id": "test_python_hard_001",
        "title": "Implement Data Processing Pipeline",
        "description": "Create a data processing pipeline with multiple functions",
        "difficulty": "hard",
        "task_category": "feature_implementation",
        "language": "python",
        "task_prompt": "Implement a data processing pipeline that includes: 1) A function to read CSV files, 2) A function to filter data by condition, 3) A function to aggregate data. Return JSON format: {\"files\": {\"pipeline.py\": \"...\"}}",
        "context_files": {
            "data_utils.py": """import csv
import json

def read_json(filepath):
    '''Read JSON file'''
    with open(filepath, 'r') as f:
        return json.load(f)

def write_json(data, filepath):
    '''Write data to JSON file'''
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)""",
            "processors.py": """class DataProcessor:
    def __init__(self):
        self.data = []
    
    def load(self, data):
        self.data = data
        return self
    
    def process(self):
        return self.data"""
        },
        "metadata": {
            "project_path": "./data/generated/test_project",
            "context_length": 1200
        }
    }
    
    # Сохранить сценарии
    scenario_file1 = scenarios_dir / "test_easy_scenario.json"
    with open(scenario_file1, 'w') as f:
        json.dump(scenario1, f, indent=2)
    
    scenario_file2 = scenarios_dir / "test_hard_scenario.json"
    with open(scenario_file2, 'w') as f:
        json.dump(scenario2, f, indent=2)
    
    print(f"✅ Создано 2 тестовых сценария:")
    print(f"  📄 {scenario_file1}")
    print(f"  📄 {scenario_file2}")
    
    return [scenario_file1, scenario_file2]


def print_instructions():
    """Вывести инструкции для запуска"""
    print("="*60)
    print("📋 ИНСТРУКЦИИ ДЛЯ ЗАПУСКА ОЦЕНКИ")
    print("="*60)
    print("\n1. Установите зависимости:")
    print("   pip install -r requirements.txt")
    print("\n2. Запустите оценку БЕЗ retrieval:")
    print("   locobench evaluate --model deepseek-ai/deepseek-coder-1.3b-instruct \\")
    print("     --task-category feature_implementation \\")
    print("     --difficulty easy \\")
    print("     --output-file evaluation_results/no_retrieval.json")
    print("\n3. Включите retrieval в config.yaml:")
    print("   retrieval:")
    print("     enabled: true")
    print("\n4. Запустите оценку С retrieval:")
    print("   locobench evaluate --model deepseek-ai/deepseek-coder-1.3b-instruct \\")
    print("     --task-category feature_implementation \\")
    print("     --difficulty hard \\")
    print("     --output-file evaluation_results/with_retrieval.json")
    print("\nИли используйте скрипт для автоматического сравнения:")
    print("   python3 compare_retrieval_full.py")
    print("="*60)


def main():
    """Главная функция"""
    print("🔧 Подготовка тестовых сценариев...")
    
    # Создать директории
    Path("data/output/scenarios").mkdir(parents=True, exist_ok=True)
    Path("evaluation_results").mkdir(exist_ok=True)
    
    # Создать тестовые сценарии
    scenario_files = create_test_scenarios()
    
    print("\n✅ Тестовые сценарии готовы!")
    print("\n" + "="*60)
    print("📝 Следующие шаги:")
    print("="*60)
    
    print_instructions()
    
    # Создать скрипт для автоматического запуска
    script_content = '''#!/usr/bin/env python3
"""
Автоматический запуск сравнения с retrieval и без него
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

def update_config_retrieval(enabled):
    """Обновить config.yaml для включения/выключения retrieval"""
    import yaml
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    config['retrieval']['enabled'] = enabled
    with open("config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def run_evaluation(difficulty, retrieval_enabled, output_suffix):
    """Запустить оценку"""
    update_config_retrieval(retrieval_enabled)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results/deepseek_{output_suffix}_{timestamp}.json"
    
    cmd = [
        sys.executable, "-m", "locobench.cli",
        "evaluate",
        "--model", "deepseek-ai/deepseek-coder-1.3b-instruct",
        "--task-category", "feature_implementation",
        "--difficulty", difficulty,
        "--output-file", output_file,
        "--no-resume"
    ]
    
    print(f"\\n🚀 {'С' if retrieval_enabled else 'БЕЗ'} retrieval: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Результаты: {output_file}")
        return output_file
    else:
        print(f"❌ Ошибка:\\n{result.stderr}")
        return None

if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        print("❌ Установите PyYAML: pip install pyyaml")
        sys.exit(1)
    
    print("🔬 Сравнение DeepSeek с retrieval и без него\\n")
    
    # БЕЗ retrieval (easy difficulty)
    file1 = run_evaluation("easy", False, "no_retrieval")
    
    # С retrieval (hard difficulty)
    file2 = run_evaluation("hard", True, "with_retrieval")
    
    if file1 and file2:
        print("\\n✅ Оба запуска завершены!")
        print(f"📊 Результаты БЕЗ retrieval: {file1}")
        print(f"📊 Результаты С retrieval: {file2}")
    else:
        print("\\n⚠️ Один или оба запуска завершились с ошибкой")
'''
    
    script_file = Path("compare_retrieval_full.py")
    with open(script_file, 'w') as f:
        f.write(script_content)
    script_file.chmod(0o755)
    
    print(f"\n✅ Создан скрипт для автоматического запуска: {script_file}")
    print(f"\n💡 Для запуска выполните:")
    print(f"   python3 {script_file}")


if __name__ == "__main__":
    main()
