#!/usr/bin/env python3
"""
Простой скрипт для сравнения результатов с retrieval и без него
Использует CLI команды locobench
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime


def run_evaluation(retrieval_enabled: bool):
    """Запустить оценку через CLI"""
    print(f"\n{'='*60}")
    print(f"🧪 Запуск оценки {'С' if retrieval_enabled else 'БЕЗ'} retrieval")
    print(f"{'='*60}\n")
    
    # Обновить config.yaml для включения/выключения retrieval
    config_path = Path("config.yaml")
    if not config_path.exists():
        print(f"❌ Файл конфигурации не найден: {config_path}")
        return None
    
    # Прочитать конфигурацию
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Обновить retrieval настройки
    config['retrieval']['enabled'] = retrieval_enabled
    
    # Временно сохранить обновленную конфигурацию
    backup_config = config_path.read_text()
    temp_config_path = Path("config_temp.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        # Запустить оценку
        model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
        output_suffix = "with_retrieval" if retrieval_enabled else "no_retrieval"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_results/deepseek_comparison_{output_suffix}_{timestamp}.json"
        
        cmd = [
            sys.executable, "-m", "locobench.cli",
            "evaluate",
            "--config-path", str(temp_config_path),
            "--model", model_name,
            "--output-file", output_file,
            "--no-resume"  # Начать заново для чистого сравнения
        ]
        
        print(f"🚀 Команда: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 час таймаут
        )
        
        if result.returncode == 0:
            print("✅ Оценка завершена успешно")
            print(f"📄 Результаты сохранены: {output_file}")
            return output_file
        else:
            print(f"❌ Ошибка при выполнении оценки:")
            print(result.stderr)
            return None
            
    finally:
        # Удалить временный файл конфигурации
        if temp_config_path.exists():
            temp_config_path.unlink()


def compare_results(file_without, file_with):
    """Сравнить результаты из двух файлов"""
    print(f"\n{'='*60}")
    print("📊 Сравнение результатов")
    print(f"{'='*60}\n")
    
    def load_results(file_path):
        if not Path(file_path).exists():
            return None
        with open(file_path, 'r') as f:
            return json.load(f)
    
    results_without = load_results(file_without)
    results_with = load_results(file_with)
    
    if not results_without:
        print(f"❌ Не удалось загрузить результаты БЕЗ retrieval: {file_without}")
        return
    
    if not results_with:
        print(f"❌ Не удалось загрузить результаты С retrieval: {file_with}")
        return
    
    # Извлечь метрики
    print("📈 Сравнение метрик:\n")
    
    # Простое сравнение - вывести основные метрики
    print("РЕЗУЛЬТАТЫ БЕЗ retrieval:")
    print(f"  Файл: {file_without}")
    if 'results' in results_without:
        print(f"  Моделей оценено: {len(results_without['results'])}")
    
    print("\nРЕЗУЛЬТАТЫ С retrieval:")
    print(f"  Файл: {file_with}")
    if 'results' in results_with:
        print(f"  Моделей оценено: {len(results_with['results'])}")
    
    # Сохранить сравнение
    comparison_file = Path("evaluation_results") / "comparison_summary.json"
    with open(comparison_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'without_retrieval_file': str(file_without),
            'with_retrieval_file': str(file_with),
            'without_retrieval': results_without,
            'with_retrieval': results_with
        }, f, indent=2)
    
    print(f"\n✅ Сравнение сохранено: {comparison_file}")


def main():
    """Главная функция"""
    print("🔬 Сравнение оценки DeepSeek с retrieval и без него")
    print("="*60)
    
    # Создать директорию для результатов
    Path("evaluation_results").mkdir(exist_ok=True)
    
    # Запуск 1: БЕЗ retrieval
    file_without = run_evaluation(retrieval_enabled=False)
    
    if not file_without:
        print("❌ Ошибка при запуске оценки БЕЗ retrieval")
        return
    
    # Запуск 2: С retrieval
    file_with = run_evaluation(retrieval_enabled=True)
    
    if not file_with:
        print("❌ Ошибка при запуске оценки С retrieval")
        return
    
    # Сравнить результаты
    compare_results(file_without, file_with)
    
    print("\n✅ Сравнение завершено!")


if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        print("❌ Необходимо установить PyYAML: pip install pyyaml")
        sys.exit(1)
    
    main()
