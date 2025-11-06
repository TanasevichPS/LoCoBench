#!/usr/bin/env python3
"""
Полный пайплайн для retrieval модели с Hugging Face
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def run_full_retrieval_pipeline():
    """Запустить полный пайплайн с retrieval"""
    print("="*60)
    print("🚀 ПОЛНЫЙ ПАЙПЛАЙН RETRIEVAL С HUGGING FACE")
    print("="*60)
    
    try:
        from locobench.core.config import Config
        from locobench.retrieval import retrieve_relevant, load_context_files_from_scenario
        from locobench.generation.synthetic_generator import MultiLLMGenerator
        from locobench.utils.llm_parsing import parse_llm_response
        
        # Загрузить конфигурацию
        config = Config.from_yaml("config.yaml")
        
        # Включить retrieval
        config.retrieval.enabled = True
        config.retrieval.method = "embedding"
        config.retrieval.model_name = "all-MiniLM-L6-v2"
        config.retrieval.top_k = 5
        
        print("\n✅ Конфигурация загружена")
        print(f"   Retrieval: {'Включен' if config.retrieval.enabled else 'Выключен'}")
        print(f"   Метод: {config.retrieval.method}")
        print(f"   Embedding модель: {config.retrieval.model_name}")
        
        # Загрузить тестовый сценарий
        scenario_file = Path("data/output/scenarios/test_hard_scenario.json")
        if not scenario_file.exists():
            print(f"❌ Сценарий не найден: {scenario_file}")
            return
        
        with open(scenario_file, 'r') as f:
            scenario = json.load(f)
        
        print(f"\n📋 Сценарий загружен: {scenario.get('id', 'unknown')}")
        print(f"   Difficulty: {scenario.get('difficulty', 'unknown')}")
        print(f"   Task: {scenario.get('title', 'unknown')}")
        
        # Извлечь контекстные файлы
        context_files = scenario.get('context_files', {})
        if isinstance(context_files, dict):
            context_files_dict = context_files
        else:
            context_files_dict = {}
        
        print(f"\n📁 Контекстные файлы: {len(context_files_dict)}")
        for filename in context_files_dict.keys():
            print(f"   - {filename}")
        
        # Шаг 1: Retrieval
        print(f"\n{'='*60}")
        print("🔍 ШАГ 1: RETRIEVAL")
        print(f"{'='*60}")
        
        task_prompt = scenario.get('task_prompt', '')
        print(f"📝 Задача: {task_prompt[:100]}...")
        
        retrieved_context = retrieve_relevant(
            context_files_dict,
            task_prompt,
            top_k=config.retrieval.top_k,
            method=config.retrieval.method,
            model_name=config.retrieval.model_name,
            chunk_size=config.retrieval.chunk_size,
            chunk_overlap=config.retrieval.chunk_overlap,
            min_similarity=config.retrieval.min_similarity
        )
        
        print(f"\n✅ Retrieval завершен")
        print(f"   Найдено релевантных фрагментов: {len(retrieved_context)} символов")
        if retrieved_context:
            print(f"   Превью: {retrieved_context[:200]}...")
        else:
            print("   ⚠️ Retrieval вернул пустой результат")
        
        # Шаг 2: Генерация с Hugging Face моделью
        print(f"\n{'='*60}")
        print("🤖 ШАГ 2: ГЕНЕРАЦИЯ С HUGGING FACE")
        print(f"{'='*60}")
        
        # Использовать небольшую модель для кода
        hf_model = "deepseek-ai/deepseek-coder-1.3b-instruct"
        print(f"   Модель: {hf_model}")
        
        # Создать промпт с retrieval контекстом
        full_prompt = f"""You are an expert Python developer. Your task is to provide a complete, working solution.

**TASK**: {scenario.get('title', 'Development Task')}

**DESCRIPTION**: {scenario.get('description', '')}

**REQUIREMENTS**: 
{task_prompt}

**RETRIEVED CONTEXT** (reference only, do not copy verbatim):
{retrieved_context if retrieved_context else 'No relevant context retrieved'}

**FULL CONTEXT FILES**: {', '.join(context_files_dict.keys())}

**CRITICAL INSTRUCTIONS**:
1. You MUST respond with valid JSON in the exact format shown below
2. Each file MUST contain complete, syntactically correct Python code
3. Use the retrieved context to understand the codebase structure
4. Do NOT truncate your response - provide the complete solution

**REQUIRED RESPONSE FORMAT**:
```json
{{
    "files": {{
        "pipeline.py": "def read_csv(filepath):\\n    import csv\\n    with open(filepath, 'r') as f:\\n        return list(csv.reader(f))"
    }}
}}
```

Generate your response now:"""
        
        print(f"\n⏳ Генерация решения (это может занять время при первой загрузке модели)...")
        
        generator = MultiLLMGenerator(config)
        
        response = await generator.generate_with_huggingface(hf_model, full_prompt)
        
        print(f"\n✅ Генерация завершена")
        print(f"   Длина ответа: {len(response)} символов")
        print(f"\n📝 Ответ модели:")
        print(f"{response[:500]}...")
        
        # Шаг 3: Парсинг ответа
        print(f"\n{'='*60}")
        print("🔧 ШАГ 3: ПАРСИНГ ОТВЕТА")
        print(f"{'='*60}")
        
        parsed_files = parse_llm_response(response, expected_language='python')
        
        if parsed_files:
            print(f"\n✅ Парсинг успешен")
            print(f"   Извлечено файлов: {len(parsed_files)}")
            for filename, content in parsed_files.items():
                print(f"\n   📄 {filename}:")
                print(f"   {'-'*40}")
                print(f"   {content[:300]}...")
                print(f"   Длина: {len(content)} символов")
        else:
            print(f"\n⚠️ Парсинг не удался")
            print(f"   Попробуем извлечь код вручную...")
            # Попробовать найти код в блоке
            if "```python" in response:
                code_start = response.find("```python") + 9
                code_end = response.find("```", code_start)
                if code_end > code_start:
                    code = response[code_start:code_end].strip()
                    print(f"   Найден код в блоке: {len(code)} символов")
        
        # Шаг 4: Сохранение результатов
        print(f"\n{'='*60}")
        print("💾 ШАГ 4: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print(f"{'='*60}")
        
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"retrieval_pipeline_results_{timestamp}.json"
        
        results = {
            'timestamp': timestamp,
            'scenario_id': scenario.get('id', 'unknown'),
            'model': hf_model,
            'retrieval': {
                'enabled': True,
                'method': config.retrieval.method,
                'embedding_model': config.retrieval.model_name,
                'top_k': config.retrieval.top_k,
                'retrieved_context_length': len(retrieved_context),
                'retrieved_context_preview': retrieved_context[:500] if retrieved_context else None,
                'chunk_size': config.retrieval.chunk_size,
                'chunk_overlap': config.retrieval.chunk_overlap,
                'min_similarity': config.retrieval.min_similarity
            },
            'generation': {
                'response_length': len(response),
                'response_preview': response[:500]
            },
            'parsing': {
                'success': parsed_files is not None,
                'files_count': len(parsed_files) if parsed_files else 0,
                'parsed_files': parsed_files if parsed_files else None
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Результаты сохранены: {results_file}")
        
        # Итоговая статистика
        print(f"\n{'='*60}")
        print("📊 ИТОГОВАЯ СТАТИСТИКА")
        print(f"{'='*60}")
        print(f"✅ Retrieval: {'Успешно' if retrieved_context else 'Не удалось'}")
        print(f"✅ Генерация: Успешно ({len(response)} символов)")
        print(f"✅ Парсинг: {'Успешно' if parsed_files else 'Не удалось'}")
        if parsed_files:
            print(f"   Извлечено файлов: {len(parsed_files)}")
            total_code = sum(len(c) for c in parsed_files.values())
            print(f"   Всего кода: {total_code} символов")
        
        print(f"\n🎉 Пайплайн завершен успешно!")
        
        return results
        
    except ImportError as e:
        print(f"\n❌ Ошибка импорта: {e}")
        print(f"   Установите зависимости: pip install transformers torch sentence-transformers")
        return None
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении пайплайна: {e}")
        import traceback
        print(traceback.format_exc())
        return None


async def main():
    """Главная функция"""
    results = await run_full_retrieval_pipeline()
    
    if results:
        print(f"\n✅ Полный пайплайн выполнен успешно!")
        print(f"📄 Результаты сохранены в evaluation_results/")
    else:
        print(f"\n❌ Пайплайн завершился с ошибкой")


if __name__ == "__main__":
    asyncio.run(main())
