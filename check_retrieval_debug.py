#!/usr/bin/env python3
"""
Диагностический скрипт для проверки работы ретривера и различий в промптах.
Проверяет:
1. Действительно ли ретривер вызывается и возвращает контент
2. Различаются ли промпты между режимами с ретривером и без
3. Нет ли кэширования на уровне модели
"""

import sys
import yaml
from pathlib import Path
from locobench.core.config import Config
from locobench.retrieval import retrieve_relevant, load_context_files_from_scenario
import json

def load_scenario(scenario_id=None):
    """Загружает первый доступный сценарий для тестирования"""
    config = Config.from_yaml("config.yaml")
    scenarios_dir = Path(config.data.output_dir) / "scenarios"
    
    if not scenarios_dir.exists():
        print(f"❌ Scenarios directory not found: {scenarios_dir}")
        sys.exit(1)
    
    # Загружаем все сценарии
    all_scenarios = []
    for scenario_file in scenarios_dir.glob("*.json"):
        with open(scenario_file, 'r') as f:
            scenario_data = json.load(f)
            all_scenarios.append(scenario_data)
    
    if not all_scenarios:
        print(f"❌ No scenarios found in {scenarios_dir}")
        sys.exit(1)
    
    # Если указан ID, ищем его, иначе берем первый hard/expert
    if scenario_id:
        for s in all_scenarios:
            if s.get('id') == scenario_id:
                return s, config
        print(f"⚠️ Scenario {scenario_id} not found, using first available")
    
    # Ищем hard или expert сценарий
    for s in all_scenarios:
        diff = s.get('difficulty', '').lower()
        if diff in ['hard', 'expert']:
            return s, config
    
    # Если не нашли, берем первый
    return all_scenarios[0], config

def check_retrieval_enabled(config_path):
    """Проверяет, включен ли ретривер в конфиге"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    enabled = config.get('retrieval', {}).get('enabled', False)
    difficulties = config.get('retrieval', {}).get('difficulties', [])
    
    print(f"📋 Config: {config_path}")
    print(f"   Retrieval enabled: {enabled}")
    print(f"   Retrieval difficulties: {difficulties}")
    print()
    
    return enabled, difficulties

def test_retrieval(scenario, config, enabled=True):
    """Тестирует работу ретривера на конкретном сценарии"""
    print("="*60)
    print(f"🧪 Testing retrieval (enabled={enabled})")
    print("="*60)
    
    scenario_id = scenario.get('id', 'unknown')
    difficulty = scenario.get('difficulty', '').lower()
    task_prompt = scenario.get('description', '') or scenario.get('title', '')
    
    print(f"Scenario ID: {scenario_id}")
    print(f"Difficulty: {difficulty}")
    print(f"Task prompt length: {len(task_prompt)} chars")
    print()
    
    retrieval_config = config.retrieval
    
    # Проверяем условия применения ретривера
    should_apply = enabled and difficulty in [d.lower() for d in retrieval_config.difficulties]
    print(f"🔍 Should apply retrieval: {should_apply}")
    print(f"   - Retrieval enabled: {enabled}")
    print(f"   - Difficulty '{difficulty}' in {retrieval_config.difficulties}: {difficulty in [d.lower() for d in retrieval_config.difficulties]}")
    print()
    
    if not should_apply:
        print("⏭️  Retrieval will NOT be applied (conditions not met)")
        return None, None
    
    # Загружаем project_dir
    project_path = scenario.get('project_path')
    project_dir = None
    if project_path:
        generated_dir = Path(config.data.generated_dir)
        project_dir = generated_dir / project_path
        if not project_dir.exists():
            print(f"⚠️  Project directory not found: {project_dir}")
            project_dir = None
    
    # Загружаем context_files
    context_obj = scenario.get('context_files')
    context_files_content = {}
    
    if isinstance(context_obj, dict):
        context_files_content = {
            path: content for path, content in context_obj.items() if isinstance(content, str)
        }
        print(f"📚 Loaded {len(context_files_content)} files from dict context_files")
    elif isinstance(context_obj, list) and project_dir:
        context_files_content = load_context_files_from_scenario(
            scenario,
            project_dir=project_dir,
            include_all_project_files=True,
        )
        print(f"📚 Loaded {len(context_files_content)} files from project directory")
    
    if not context_files_content:
        print("⚠️  No context files available for retrieval")
        return None, None
    
    print(f"📊 Total context files: {len(context_files_content)}")
    total_size = sum(len(content) for content in context_files_content.values())
    print(f"📊 Total context size: {total_size:,} chars")
    print()
    
    # Вызываем ретривер
    print("🔍 Calling retrieve_relevant()...")
    try:
        retrieved_context = retrieve_relevant(
            context_files_content,
            task_prompt,
            top_k=retrieval_config.top_k,
            method=retrieval_config.method,
            model_name=retrieval_config.model_name,
            project_dir=project_dir,
            top_percent=retrieval_config.top_percent,
            max_context_tokens=retrieval_config.max_context_tokens,
            local_model_path=retrieval_config.local_model_path,
            chunk_size=retrieval_config.chunk_size,
            smart_chunking=getattr(retrieval_config, 'smart_chunking', True),
            chunks_per_file=getattr(retrieval_config, 'chunks_per_file', 5),
            retrieval_chunk_size=getattr(retrieval_config, 'retrieval_chunk_size', 2000),
        )
        
        if retrieved_context:
            print(f"✅ Retrieval SUCCESS")
            print(f"   Retrieved context length: {len(retrieved_context):,} chars")
            print(f"   Reduction: {100 * (1 - len(retrieved_context) / total_size):.1f}%")
            
            # Показываем первые 500 символов
            preview = retrieved_context[:500]
            print(f"\n📄 Preview (first 500 chars):")
            print("-" * 60)
            print(preview)
            if len(retrieved_context) > 500:
                print("...")
            print("-" * 60)
            
            return retrieved_context, context_files_content
        else:
            print("❌ Retrieval returned EMPTY result")
            return None, context_files_content
            
    except Exception as e:
        print(f"❌ Retrieval FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, context_files_content

def compare_prompts(scenario, config, with_retrieval_context, without_retrieval_context):
    """Сравнивает промпты с ретривером и без"""
    print("\n" + "="*60)
    print("📊 Comparing prompts")
    print("="*60)
    
    task_prompt = scenario.get('description', '') or scenario.get('title', '')
    
    # Создаем промпты как в evaluator
    if with_retrieval_context:
        context_section_with = f"""**RETRIEVED CONTEXT** (use this for reasoning - most relevant code fragments):
{with_retrieval_context}

**FULL CONTEXT FILES**: {', '.join(scenario.get('context_files', []))}
"""
    else:
        context_section_with = "**RETRIEVED CONTEXT**: (empty - retrieval disabled)"
    
    # Без ретривера - используем полные файлы
    context_obj = scenario.get('context_files')
    if isinstance(context_obj, dict):
        context_files_content = {
            path: content for path, content in context_obj.items() if isinstance(content, str)
        }
        # Формируем секцию как в evaluator (упрощенно)
        context_section_without = f"**CONTEXT FILES**: {len(context_files_content)} files loaded"
    else:
        context_section_without = f"**CONTEXT FILES**: {len(context_obj) if isinstance(context_obj, list) else 0} files"
    
    prompt_with = f"""**TASK**: {scenario.get('title', 'Development Task')}

**DESCRIPTION**: {scenario.get('description', '')}

{context_section_with}
"""
    
    prompt_without = f"""**TASK**: {scenario.get('title', 'Development Task')}

**DESCRIPTION**: {scenario.get('description', '')}

{context_section_without}
"""
    
    print(f"📏 Prompt WITH retrieval: {len(prompt_with):,} chars")
    print(f"📏 Prompt WITHOUT retrieval: {len(prompt_without):,} chars")
    print(f"📊 Difference: {len(prompt_with) - len(prompt_without):,} chars")
    print()
    
    # Показываем различия
    print("🔍 Key differences:")
    if with_retrieval_context:
        print("   ✅ WITH retrieval: Uses RETRIEVED CONTEXT section")
        print(f"      - Retrieved context: {len(with_retrieval_context):,} chars")
    else:
        print("   ❌ WITH retrieval: Empty (retrieval failed or disabled)")
    
    print("   📋 WITHOUT retrieval: Uses full context files")
    print()
    
    # Проверяем, действительно ли есть разница
    if with_retrieval_context and len(with_retrieval_context) > 0:
        print("✅ Prompts ARE different - retrieval is working")
        return True
    else:
        print("⚠️  Prompts are similar - retrieval may not be working correctly")
        return False

def main():
    if len(sys.argv) > 1:
        scenario_id = sys.argv[1]
    else:
        scenario_id = None
    
    print("="*60)
    print("🔍 Retrieval Diagnostic Tool")
    print("="*60)
    print()
    
    # Загружаем сценарий
    scenario, config = load_scenario(scenario_id)
    print(f"📁 Loaded scenario: {scenario.get('id', 'unknown')}")
    print()
    
    # Проверяем конфиги
    print("Checking configurations...")
    config_with = Path("config_with_retrieval_test.yaml")
    config_without = Path("config_without_retrieval_test.yaml")
    
    # Создаем временные конфиги
    base_config = Config.from_yaml("config.yaml")
    
    # Конфиг с ретривером
    config_dict_with = base_config.to_dict()
    config_dict_with['retrieval']['enabled'] = True
    with open(config_with, 'w') as f:
        yaml.dump(config_dict_with, f, default_flow_style=False)
    
    # Конфиг без ретривера
    config_dict_without = base_config.to_dict()
    config_dict_without['retrieval']['enabled'] = False
    with open(config_without, 'w') as f:
        yaml.dump(config_dict_without, f, default_flow_style=False)
    
    enabled_with, difficulties_with = check_retrieval_enabled(config_with)
    enabled_without, difficulties_without = check_retrieval_enabled(config_without)
    
    # Тестируем ретривер с включенным режимом
    config_with_obj = Config.from_yaml(str(config_with))
    retrieved_with, context_files_with = test_retrieval(scenario, config_with_obj, enabled=True)
    
    print("\n")
    
    # Тестируем ретривер с выключенным режимом
    config_without_obj = Config.from_yaml(str(config_without))
    retrieved_without, context_files_without = test_retrieval(scenario, config_without_obj, enabled=False)
    
    # Сравниваем промпты
    are_different = compare_prompts(scenario, config_with_obj, retrieved_with, retrieved_without)
    
    # Итоги
    print("\n" + "="*60)
    print("📋 Summary")
    print("="*60)
    print(f"✅ Retrieval enabled config: Retrieval {'WORKING' if retrieved_with else 'FAILED/EMPTY'}")
    print(f"✅ Retrieval disabled config: Retrieval {'DISABLED' if not retrieved_without else 'UNEXPECTEDLY ACTIVE'}")
    print(f"✅ Prompts are different: {'YES' if are_different else 'NO'}")
    print()
    
    if retrieved_with and are_different:
        print("✅ Retrieval is working correctly!")
    else:
        print("⚠️  Potential issues detected:")
        if not retrieved_with:
            print("   - Retrieval returns empty result even when enabled")
        if not are_different:
            print("   - Prompts are not different between modes")
    
    # Очистка
    config_with.unlink(missing_ok=True)
    config_without.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
