#!/usr/bin/env python3
"""
Упрощенный полный пайплайн для retrieval модели с Hugging Face
Работает напрямую без полной инициализации LoCoBench
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def run_simple_retrieval_pipeline():
    """Упрощенный пайплайн с retrieval и Hugging Face"""
    print("="*60)
    print("🚀 ПОЛНЫЙ ПАЙПЛАЙН RETRIEVAL С HUGGING FACE")
    print("   (Упрощенная версия)")
    print("="*60)
    
    try:
        # Импорт необходимых модулей
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        import numpy as np
        
        # Загрузить сценарий
        scenario_file = Path("data/output/scenarios/test_hard_scenario.json")
        if not scenario_file.exists():
            print(f"❌ Сценарий не найден: {scenario_file}")
            return None
        
        with open(scenario_file, 'r') as f:
            scenario = json.load(f)
        
        print(f"\n✅ Сценарий загружен: {scenario.get('id', 'unknown')}")
        print(f"   Difficulty: {scenario.get('difficulty', 'unknown')}")
        print(f"   Task: {scenario.get('title', 'unknown')}")
        
        context_files = scenario.get('context_files', {})
        task_prompt = scenario.get('task_prompt', '')
        
        print(f"\n📁 Контекстные файлы: {len(context_files)}")
        
        # ШАГ 1: RETRIEVAL
        print(f"\n{'='*60}")
        print("🔍 ШАГ 1: RETRIEVAL")
        print(f"{'='*60}")
        
        print("📦 Загрузка embedding модели...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding модель загружена")
        
        # Разделить код на чанки
        def split_code(code, chunk_size=512):
            lines = code.split('\n')
            chunks = []
            current_chunk = []
            current_size = 0
            
            for line in lines:
                line_size = len(line) + 1
                if current_size + line_size > chunk_size and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_size
                else:
                    current_chunk.append(line)
                    current_size += line_size
            
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            
            return chunks
        
        # Подготовить чанки
        all_chunks = []
        chunk_info = []
        
        for filename, code_content in context_files.items():
            file_chunks = split_code(code_content)
            for idx, chunk in enumerate(file_chunks):
                chunk_info.append((filename, idx, chunk))
                all_chunks.append(chunk)
        
        print(f"   Создано {len(all_chunks)} чанков из {len(context_files)} файлов")
        
        # Вычислить эмбеддинги
        print("   Вычисление эмбеддингов...")
        all_texts = all_chunks + [task_prompt]
        embeddings = embedding_model.encode(all_texts, show_progress_bar=False)
        
        query_embedding = embeddings[-1]
        chunk_embeddings = embeddings[:-1]
        
        # Вычислить косинусное сходство
        query_norm = np.linalg.norm(query_embedding)
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
        similarities = np.dot(chunk_embeddings, query_embedding) / (chunk_norms * query_norm)
        similarities = np.nan_to_num(similarities, nan=0.0)
        
        # Получить top-K
        top_k = 5
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Собрать релевантные фрагменты
        retrieved_parts = []
        for idx in top_indices:
            file_path, chunk_idx, chunk_content = chunk_info[idx]
            similarity_score = similarities[idx]
            retrieved_parts.append(
                f"From {file_path} (chunk {chunk_idx + 1}, similarity: {similarity_score:.3f}):\n{chunk_content}"
            )
        
        retrieved_context = "\n\n".join(retrieved_parts)
        
        print(f"✅ Retrieval завершен")
        print(f"   Найдено релевантных фрагментов: {len(retrieved_context)} символов")
        print(f"   Top-K: {top_k}")
        if retrieved_context:
            print(f"   Превью: {retrieved_context[:200]}...")
        
        # ШАГ 2: ГЕНЕРАЦИЯ С HUGGING FACE
        print(f"\n{'='*60}")
        print("🤖 ШАГ 2: ГЕНЕРАЦИЯ С HUGGING FACE")
        print(f"{'='*60}")
        
        hf_model = "deepseek-ai/deepseek-coder-1.3b-instruct"
        print(f"   Модель: {hf_model}")
        print(f"   ⏳ Загрузка модели (это может занять время при первом запуске)...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Устройство: {device}")
        
        # Загрузить токенизатор и модель
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            hf_model,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
        print(f"✅ Модель загружена")
        
        # Создать промпт
        full_prompt = f"""You are an expert Python developer. Provide a complete solution.

**TASK**: {scenario.get('title', 'Development Task')}

**DESCRIPTION**: {scenario.get('description', '')}

**REQUIREMENTS**: 
{task_prompt}

**RETRIEVED CONTEXT** (reference only, do not copy verbatim):
{retrieved_context if retrieved_context else 'No relevant context retrieved'}

**FULL CONTEXT FILES**: {', '.join(context_files.keys())}

**INSTRUCTIONS**:
1. Respond with valid JSON format
2. Provide complete Python code
3. Use the retrieved context to understand the codebase structure

**REQUIRED FORMAT**:
```json
{{
    "files": {{
        "pipeline.py": "def read_csv(filepath):\\n    import csv\\n    with open(filepath, 'r') as f:\\n        return list(csv.reader(f))"
    }}
}}
```

Generate your response:"""
        
        print(f"\n   Генерация ответа...")
        
        # Токенизация
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        
        # Генерация
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Декодирование
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлечь только новую часть
        if full_prompt in generated_text:
            response = generated_text.split(full_prompt, 1)[1].strip()
        else:
            response = generated_text[len(full_prompt):].strip()
        
        print(f"✅ Генерация завершена")
        print(f"   Длина ответа: {len(response)} символов")
        print(f"\n📝 Ответ модели:")
        print(f"{response}")
        
        # ШАГ 3: ПАРСИНГ
        print(f"\n{'='*60}")
        print("🔧 ШАГ 3: ПАРСИНГ ОТВЕТА")
        print(f"{'='*60}")
        
        # Простой парсинг JSON
        parsed_files = None
        
        # Попробовать найти JSON в ответе
        import re
        json_patterns = [
            r'```json\s*\n?(.*?)\n?\s*```',
            r'```\s*\n?(.*?)\n?\s*```',
            r'(\{.*?"files".*?\})',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    if 'files' in data and isinstance(data['files'], dict):
                        parsed_files = data['files']
                        print(f"✅ JSON найден и распарсен")
                        break
                except:
                    continue
        
        if not parsed_files:
            # Попробовать найти код напрямую
            if "```python" in response:
                code_start = response.find("```python") + 9
                code_end = response.find("```", code_start)
                if code_end > code_start:
                    code = response[code_start:code_end].strip()
                    parsed_files = {"extracted.py": code}
                    print(f"✅ Код извлечен из блока")
        
        if parsed_files:
            print(f"\n✅ Парсинг успешен")
            print(f"   Извлечено файлов: {len(parsed_files)}")
            for filename, content in parsed_files.items():
                print(f"\n   📄 {filename}:")
                print(f"   {'-'*40}")
                print(f"   {content[:400]}...")
                print(f"   Длина: {len(content)} символов")
        else:
            print(f"\n⚠️ Парсинг не удался")
            print(f"   Полный ответ сохранен для анализа")
        
        # ШАГ 4: СОХРАНЕНИЕ
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
            'device': device,
            'retrieval': {
                'enabled': True,
                'embedding_model': 'all-MiniLM-L6-v2',
                'top_k': top_k,
                'chunks_processed': len(all_chunks),
                'retrieved_context_length': len(retrieved_context),
                'retrieved_context': retrieved_context[:1000] if retrieved_context else None
            },
            'generation': {
                'response_length': len(response),
                'response': response
            },
            'parsing': {
                'success': parsed_files is not None,
                'files_count': len(parsed_files) if parsed_files else 0,
                'parsed_files': parsed_files if parsed_files else None
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Результаты сохранены: {results_file}")
        
        # ИТОГИ
        print(f"\n{'='*60}")
        print("📊 ИТОГОВАЯ СТАТИСТИКА")
        print(f"{'='*60}")
        print(f"✅ Retrieval: Успешно ({len(retrieved_context)} символов)")
        print(f"✅ Генерация: Успешно ({len(response)} символов)")
        print(f"✅ Парсинг: {'Успешно' if parsed_files else 'Частично'}")
        if parsed_files:
            total_code = sum(len(c) for c in parsed_files.values())
            print(f"   Извлечено файлов: {len(parsed_files)}")
            print(f"   Всего кода: {total_code} символов")
        
        print(f"\n🎉 Пайплайн завершен успешно!")
        print(f"📄 Полные результаты: {results_file}")
        
        return results
        
    except ImportError as e:
        print(f"\n❌ Ошибка импорта: {e}")
        print(f"   Установите: pip install transformers torch sentence-transformers numpy")
        return None
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        print(traceback.format_exc())
        return None


async def main():
    """Главная функция"""
    results = await run_simple_retrieval_pipeline()
    
    if results:
        print(f"\n✅ Полный пайплайн выполнен успешно!")
    else:
        print(f"\n❌ Пайплайн завершился с ошибкой")


if __name__ == "__main__":
    asyncio.run(main())
