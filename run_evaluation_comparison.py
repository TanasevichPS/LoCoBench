#!/usr/bin/env python3
"""
Скрипт для последовательного запуска оценок: сначала с ритривером, потом без.
Результаты и чекпоинты сохраняются в разных папках для полной изоляции.
"""

import subprocess
import sys
import shutil
from pathlib import Path
from datetime import datetime
import yaml

def create_config_with_retrieval(base_config_path, output_path, timestamp):
    """Создает конфиг с включенным ритривером"""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Включаем ритривер
    config['retrieval']['enabled'] = True
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created config with retrieval: {output_path}")
    print(f"   Retrieval enabled: {config['retrieval']['enabled']}")
    print(f"   Output dir: {config['data']['output_dir']}")

def create_config_without_retrieval(base_config_path, output_path, timestamp):
    """Создает конфиг с отключенным ритривером"""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Отключаем ритривер
    config['retrieval']['enabled'] = False

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created config without retrieval: {output_path}")
    print(f"   Retrieval enabled: {config['retrieval']['enabled']}")
    print(f"   Output dir: {config['data']['output_dir']}")

def cleanup_intermediate_results(config_path):
    """Удаляет папку intermediate_results из конфига и абсолютный путь перед запуском"""
    cleaned_paths = []
    
    # 1. Очищаем папку из конфига (относительный путь)
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = Path(config['data']['output_dir'])
        intermediate_dir = output_dir / "intermediate_results"
        
        if intermediate_dir.exists():
            print(f"🧹 Cleaning up intermediate_results (from config): {intermediate_dir}")
            shutil.rmtree(intermediate_dir)
            cleaned_paths.append(str(intermediate_dir))
            print(f"   ✅ Removed: {intermediate_dir}")
        else:
            print(f"   ℹ️  No intermediate_results found (from config): {intermediate_dir}")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not clean intermediate_results from config: {e}")
    
    # 2. Очищаем абсолютный путь (может быть жестко закодирован)
    absolute_intermediate_paths = [
        Path("/srv/nfs/VESO/home/polina/trsh/LoCoBench/intermediate_results"),
        Path("./intermediate_results"),
    ]
    
    for abs_path in absolute_intermediate_paths:
        if abs_path.exists():
            print(f"🧹 Cleaning up intermediate_results (absolute path): {abs_path}")
            try:
                shutil.rmtree(abs_path)
                cleaned_paths.append(str(abs_path))
                print(f"   ✅ Removed: {abs_path}")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not remove {abs_path}: {e}")
    
    if not cleaned_paths:
        print(f"   ℹ️  No intermediate_results directories found to clean")

def run_evaluation(config_path, model, output_file, description):
    """Запускает оценку с указанными параметрами"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Model: {model}")
    print(f"Output: {output_file}")
    print()
    
    # Очищаем чекпоинты перед запуском
    print("🧹 Cleaning up checkpoints before evaluation...")
    cleanup_intermediate_results(config_path)
    print()
    
    cmd = [
        "locobench", "evaluate",
        "--config-path", str(config_path),
        "--model", model,
        "--output-file", str(output_file),
        "--no-resume"  # Игнорируем любые существующие чекпоинты
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode == 0:
        print(f"\n✅ {description} completed successfully!")
        print(f"   Results saved to: {output_file}")
        return True
    else:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False

def main():
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        model = "DeepSeekR1-70B-LRI"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_config = Path("config.yaml")
    
    if not base_config.exists():
        print(f"❌ Error: {base_config} not found!")
        sys.exit(1)
    
    print("="*60)
    print("LoCoBench Evaluation Comparison")
    print("="*60)
    print(f"Model: {model}")
    print(f"Timestamp: {timestamp}")
    print()
    
    # Создаем директории для результатов
    results_with = Path(f"evaluation_results/with_retrieval_{timestamp}")
    results_without = Path(f"evaluation_results/without_retrieval_{timestamp}")
    results_with.mkdir(parents=True, exist_ok=True)
    results_without.mkdir(parents=True, exist_ok=True)
    
    # Создаем временные конфиги
    config_with = Path(f"config_with_retrieval_{timestamp}.yaml")
    config_without = Path(f"config_without_retrieval_{timestamp}.yaml")
    
    print("📝 Creating temporary configurations...")
    create_config_with_retrieval(base_config, config_with, timestamp)
    create_config_without_retrieval(base_config, config_without, timestamp)
    print()
    

    # ШАГ 2: Оценка без ритривера
    output_without = results_without / "evaluation_results.json"
    success_without = run_evaluation(
        config_without,
        model,
        output_without,
        "STEP 2: Evaluation WITHOUT Retrieval"
    )
    
    if not success_without:
        print("\n❌ Evaluation without retrieval failed.")
        sys.exit(1)
    

    print("\n⏳ Waiting 5 seconds before next evaluation...")
    import time
    time.sleep(5)


    # ШАГ 1: Оценка с ритривером
    output_with = results_with / "evaluation_results.json"
    success_with = run_evaluation(
        config_with,
        model,
        output_with,
        "STEP 1: Evaluation WITH Retrieval"
    )
    
    if not success_with:
        print("\n❌ Evaluation with retrieval failed. Stopping.")
        sys.exit(1)

    
    # Итоги
    print("\n" + "="*60)
    print("Evaluation Comparison Complete!")
    print("="*60)
    print()
    print("📊 Results:")
    print(f"   WITH retrieval:    {output_with}")
    print(f"   WITHOUT retrieval:  {output_without}")
    print()
    print("💾 Checkpoints (isolated):")
    print(f"   WITH retrieval:    data/output_with_retrieval_{timestamp}/intermediate_results/")
    print(f"   WITHOUT retrieval:  data/output_without_retrieval_{timestamp}/intermediate_results/")
    print()
    print("🧹 Temporary configs (can be deleted):")
    print(f"   {config_with}")
    print(f"   {config_without}")
    print()

if __name__ == "__main__":
    main()
