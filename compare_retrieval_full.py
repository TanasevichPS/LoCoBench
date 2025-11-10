#!/usr/bin/env python3
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
    
    print(f"\n🚀 {'С' if retrieval_enabled else 'БЕЗ'} retrieval: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Результаты: {output_file}")
        return output_file
    else:
        print(f"❌ Ошибка:\n{result.stderr}")
        return None

if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        print("❌ Установите PyYAML: pip install pyyaml")
        sys.exit(1)
    
    print("🔬 Сравнение DeepSeek с retrieval и без него\n")
    
    # БЕЗ retrieval (easy difficulty)
    file1 = run_evaluation("easy", False, "no_retrieval")
    
    # С retrieval (hard difficulty)
    file2 = run_evaluation("hard", True, "with_retrieval")
    
    if file1 and file2:
        print("\n✅ Оба запуска завершены!")
        print(f"📊 Результаты БЕЗ retrieval: {file1}")
        print(f"📊 Результаты С retrieval: {file2}")
    else:
        print("\n⚠️ Один или оба запуска завершились с ошибкой")
