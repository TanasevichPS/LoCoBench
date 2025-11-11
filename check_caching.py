#!/usr/bin/env python3
"""
Скрипт для проверки кэширования модели.
Проверяет, не кэширует ли модель результаты на основе промпта.
"""

import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
import yaml

def hash_prompt(prompt):
    """Создает хэш промпта для проверки уникальности"""
    return hashlib.md5(prompt.encode()).hexdigest()[:16]

def analyze_evaluation_results(results_file):
    """Анализирует результаты оценки на предмет кэширования"""
    print("="*60)
    print("🔍 Analyzing evaluation results for caching")
    print("="*60)
    
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Извлекаем результаты по моделям
    model_results = {}
    if isinstance(results, dict):
        if 'results' in results:
            model_results = results['results']
        else:
            model_results = results
    
    print(f"📊 Found results for {len(model_results)} model(s)")
    print()
    
    for model_name, scenarios in model_results.items():
        print(f"🤖 Model: {model_name}")
        print(f"   Scenarios: {len(scenarios)}")
        
        # Анализируем промпты и ответы
        prompt_hashes = {}
        duplicate_prompts = {}
        similar_responses = {}
        
        for scenario_result in scenarios:
            scenario_id = scenario_result.get('scenario_id', 'unknown')
            prompt = scenario_result.get('prompt', '')
            response = scenario_result.get('response', '')
            
            if prompt:
                prompt_hash = hash_prompt(prompt)
                
                if prompt_hash in prompt_hashes:
                    # Дубликат промпта
                    if prompt_hash not in duplicate_prompts:
                        duplicate_prompts[prompt_hash] = []
                    duplicate_prompts[prompt_hash].append({
                        'scenario_id': scenario_id,
                        'prompt_preview': prompt[:200],
                        'response_preview': response[:200] if response else None,
                    })
                else:
                    prompt_hashes[prompt_hash] = scenario_id
        
        print(f"   Unique prompts: {len(prompt_hashes)}")
        print(f"   Duplicate prompts: {len(duplicate_prompts)}")
        
        if duplicate_prompts:
            print(f"\n   ⚠️  Found {len(duplicate_prompts)} duplicate prompt(s):")
            for hash_val, occurrences in list(duplicate_prompts.items())[:5]:
                print(f"      Hash {hash_val}: {len(occurrences)} occurrences")
                for occ in occurrences[:2]:
                    print(f"         - Scenario: {occ['scenario_id']}")
                    print(f"           Prompt preview: {occ['prompt_preview'][:100]}...")
        
        # Проверяем идентичные ответы
        response_hashes = {}
        identical_responses = {}
        
        for scenario_result in scenarios:
            scenario_id = scenario_result.get('scenario_id', 'unknown')
            response = scenario_result.get('response', '')
            
            if response:
                response_hash = hash_prompt(response)
                
                if response_hash in response_hashes:
                    if response_hash not in identical_responses:
                        identical_responses[response_hash] = []
                    identical_responses[response_hash].append(scenario_id)
                else:
                    response_hashes[response_hash] = scenario_id
        
        print(f"   Unique responses: {len(response_hashes)}")
        print(f"   Identical responses: {len(identical_responses)}")
        
        if identical_responses:
            print(f"\n   ⚠️  Found {len(identical_responses)} identical response(s):")
            for hash_val, scenario_ids in list(identical_responses.items())[:5]:
                print(f"      Hash {hash_val}: {len(scenario_ids)} scenarios")
                print(f"         Scenarios: {', '.join(scenario_ids[:5])}")
        
        print()

def compare_two_results(file1, file2):
    """Сравнивает два файла результатов на предмет идентичности"""
    print("="*60)
    print("🔍 Comparing two result files")
    print("="*60)
    
    def load_results(f):
        if not Path(f).exists():
            return None
        with open(f, 'r') as file:
            return json.load(file)
    
    results1 = load_results(file1)
    results2 = load_results(file2)
    
    if not results1:
        print(f"❌ File 1 not found: {file1}")
        return
    if not results2:
        print(f"❌ File 2 not found: {file2}")
        return
    
    print(f"📁 File 1: {file1}")
    print(f"📁 File 2: {file2}")
    print()
    
    # Извлекаем результаты
    def extract_results(data):
        if isinstance(data, dict):
            if 'results' in data:
                return data['results']
            return data
        return data
    
    results1_dict = extract_results(results1)
    results2_dict = extract_results(results2)
    
    # Сравниваем по моделям
    all_models = set(results1_dict.keys()) | set(results2_dict.keys())
    
    for model_name in all_models:
        scenarios1 = results1_dict.get(model_name, [])
        scenarios2 = results2_dict.get(model_name, [])
        
        print(f"🤖 Model: {model_name}")
        print(f"   File 1 scenarios: {len(scenarios1)}")
        print(f"   File 2 scenarios: {len(scenarios2)}")
        
        # Создаем словари по scenario_id
        dict1 = {s.get('scenario_id'): s for s in scenarios1}
        dict2 = {s.get('scenario_id'): s for s in scenarios2}
        
        common_scenarios = set(dict1.keys()) & set(dict2.keys())
        print(f"   Common scenarios: {len(common_scenarios)}")
        
        identical_responses = 0
        different_responses = 0
        
        for scenario_id in common_scenarios:
            resp1 = dict1[scenario_id].get('response', '')
            resp2 = dict2[scenario_id].get('response', '')
            
            if resp1 == resp2:
                identical_responses += 1
            else:
                different_responses += 1
        
        print(f"   Identical responses: {identical_responses}")
        print(f"   Different responses: {different_responses}")
        
        if identical_responses > 0:
            similarity = identical_responses / len(common_scenarios) * 100
            print(f"   ⚠️  Response similarity: {similarity:.1f}%")
            
            if similarity > 50:
                print(f"   🚨 HIGH similarity detected - possible caching!")
        
        print()

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python check_caching.py <results_file>                    # Analyze single file")
        print("  python check_caching.py <file1> <file2>                   # Compare two files")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # Анализ одного файла
        results_file = sys.argv[1]
        analyze_evaluation_results(results_file)
    else:
        # Сравнение двух файлов
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        compare_two_results(file1, file2)

if __name__ == "__main__":
    main()
