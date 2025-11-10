#!/usr/bin/env python3
"""
Финальный скрипт для запуска сравнения DeepSeek с retrieval и без него
"""

import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def update_config_retrieval(enabled):
    """Обновить config.yaml для включения/выключения retrieval"""
    try:
        import yaml
    except ImportError:
        print("❌ Установите PyYAML: pip install pyyaml")
        return False
    
    config_path = Path("config.yaml")
    if not config_path.exists():
        print(f"❌ Файл конфигурации не найден: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['retrieval']['enabled'] = enabled
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Конфигурация обновлена: retrieval.enabled = {enabled}")
    return True

def run_evaluation(difficulty, retrieval_enabled, output_suffix):
    """Запустить оценку"""
    print(f"\n{'='*60}")
    print(f"🧪 Запуск оценки {'С' if retrieval_enabled else 'БЕЗ'} retrieval")
    print(f"   Difficulty: {difficulty}")
    print(f"{'='*60}\n")
    
    # Обновить конфигурацию
    if not update_config_retrieval(retrieval_enabled):
        return None
    
    # Создать директорию для результатов
    Path("evaluation_results").mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results/deepseek_{output_suffix}_{timestamp}.json"
    
    cmd = [
        sys.executable, "-m", "locobench.cli",
        "evaluate",
        "--config-path", "config.yaml",
        "--model", "deepseek-ai/deepseek-coder-1.3b-instruct",
        "--task-category", "feature_implementation",
        "--difficulty", difficulty,
        "--output-file", output_file,
        "--no-resume"
    ]
    
    print(f"🚀 Команда: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 час таймаут
        )
        
        print(f"📤 stdout:\\n{result.stdout[:500]}")
        if result.stderr:
            print(f"⚠️ stderr:\\n{result.stderr[:500]}")
        
        if result.returncode == 0:
            print(f"✅ Оценка завершена успешно")
            print(f"📄 Результаты сохранены: {output_file}")
            return output_file
        else:
            print(f"❌ Ошибка при выполнении (код: {result.returncode})")
            if result.stderr:
                print(f"Детали ошибки:\\n{result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("⏰ Таймаут при выполнении оценки (>1 час)")
        return None
    except Exception as e:
        print(f"❌ Исключение при выполнении: {e}")
        return None

def compare_results(file_without, file_with):
    """Сравнить результаты из двух файлов"""
    print(f"\n{'='*60}")
    print("📊 Сравнение результатов")
    print(f"{'='*60}\n")
    
    def load_results(file_path):
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️ Файл не найден: {file_path}")
            return None
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"❌ Ошибка при загрузке {file_path}: {e}")
            return None
    
    results_without = load_results(file_without)
    results_with = load_results(file_with)
    
    if not results_without and not results_with:
        print("❌ Не удалось загрузить результаты сравнения")
        return
    
    # Сохранить сравнение
    comparison_file = Path("evaluation_results") / "comparison_summary.json"
    
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'model': 'deepseek-ai/deepseek-coder-1.3b-instruct',
        'without_retrieval_file': str(file_without) if file_without else None,
        'with_retrieval_file': str(file_with) if file_with else None,
        'without_retrieval': results_without,
        'with_retrieval': results_with
    }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"✅ Сравнение сохранено: {comparison_file}")
    
    # Вывести краткую статистику если есть
    if results_without and 'results' in results_without:
        print(f"\\n📈 БЕЗ retrieval:")
        print(f"  Моделей оценено: {len(results_without.get('results', {}))}")
    
    if results_with and 'results' in results_with:
        print(f"\\n📈 С retrieval:")
        print(f"  Моделей оценено: {len(results_with.get('results', {}))}")

def main():
    """Главная функция"""
    print("="*60)
    print("🔬 Сравнение оценки DeepSeek с retrieval и без него")
    print("="*60)
    
    # Проверка зависимостей
    try:
        import yaml
    except ImportError:
        print("❌ Необходимо установить PyYAML: pip install pyyaml")
        sys.exit(1)
    
    # Запуск 1: БЕЗ retrieval (easy difficulty)
    file_without = run_evaluation("easy", False, "no_retrieval")
    
    if not file_without:
        print("\\n⚠️ Запуск БЕЗ retrieval не удался. Продолжаем со вторым запуском...")
    
    # Небольшая пауза между запусками
    import time
    print("\\n⏳ Пауза 5 секунд перед следующим запуском...")
    time.sleep(5)
    
    # Запуск 2: С retrieval (hard difficulty)
    file_with = run_evaluation("hard", True, "with_retrieval")
    
    if not file_with:
        print("\\n⚠️ Запуск С retrieval не удался.")
    
    # Сравнить результаты
    if file_without or file_with:
        compare_results(file_without, file_with)
        print("\\n✅ Процесс сравнения завершен!")
        print("\\n📄 Проверьте файлы результатов в evaluation_results/")
    else:
        print("\\n❌ Оба запуска завершились с ошибкой")
        print("\\n💡 Проверьте:")
        print("   1. Установлены ли зависимости: pip install -r requirements.txt")
        print("   2. Доступна ли модель: deepseek-ai/deepseek-coder-1.3b-instruct")
        print("   3. Есть ли сценарии в data/output/scenarios/")

if __name__ == "__main__":
    main()
