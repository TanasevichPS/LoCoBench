# Методы улучшения ритривера для достижения Total Score 2.3+

## Проблема: Ритривер работает как "тупая обрезка"

Текущий ритривер просто берет топ-N файлов по similarity, что не намного лучше чем просто взять первые N файлов. Нужны более продвинутые техники.

---

## 1. RE-RANKING С CROSS-ENCODER (Высокий приоритет)

### Описание:
Использовать более мощную модель для переранжирования результатов после первоначального отбора.

### Реализация:
```python
# После получения топ-50 файлов от bi-encoder, переранжировать топ-20 с cross-encoder
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# Переранжировать пары (query, document)
scores = cross_encoder.predict([(task_prompt, file_content) for file_content in top_files])
```

### Ожидаемый эффект: +0.10-0.15 балла
### Сложность: Средняя
### Приоритет: ⭐⭐⭐⭐⭐

---

## 2. QUERY EXPANSION (Высокий приоритет)

### Описание:
Расширить запрос синонимами, связанными терминами, и ключевыми словами из контекста задачи.

### Реализация:
```python
def expand_query(task_prompt):
    # Извлечь ключевые слова
    keywords = extract_keywords(task_prompt)
    # Добавить синонимы для архитектурных терминов
    synonyms = {
        'merge': ['combine', 'integrate', 'consolidate'],
        'refactor': ['restructure', 'reorganize', 'redesign'],
        'sync': ['synchronize', 'coordinate', 'align']
    }
    # Добавить связанные термины из domain knowledge
    expanded = task_prompt + " " + " ".join(expanded_terms)
    return expanded
```

### Ожидаемый эффект: +0.05-0.10 балла
### Сложность: Низкая
### Приоритет: ⭐⭐⭐⭐⭐

---

## 3. HIERARCHICAL RETRIEVAL (Средний приоритет)

### Описание:
Многоуровневая стратегия: сначала найти модули/пакеты, потом файлы внутри них.

### Реализация:
```python
# Level 1: Найти релевантные пакеты/модули
relevant_packages = find_relevant_packages(task_prompt)
# Level 2: Найти файлы внутри релевантных пакетов
files_in_packages = find_files_in_packages(relevant_packages)
# Level 3: Ранжировать файлы внутри пакетов
ranked_files = rank_files_within_packages(files_in_packages, task_prompt)
```

### Ожидаемый эффект: +0.05-0.08 балла
### Сложность: Средняя
### Приоритет: ⭐⭐⭐⭐

---

## 4. GRAPH-BASED RETRIEVAL (Высокий приоритет)

### Описание:
Использовать граф зависимостей для нахождения связанных файлов, даже если они не семантически похожи.

### Реализация:
```python
# Построить граф зависимостей
dependency_graph = build_dependency_graph(all_files)
# Найти семантически релевантные файлы
semantic_files = find_semantic_matches(task_prompt)
# Расширить через граф: найти файлы на расстоянии 1-2 шага
expanded_files = expand_via_graph(semantic_files, dependency_graph, max_depth=2)
# Ранжировать расширенный набор
ranked = rank_files(expanded_files, task_prompt)
```

### Ожидаемый эффект: +0.08-0.12 балла
### Сложность: Средняя
### Приоритет: ⭐⭐⭐⭐⭐

---

## 5. HYBRID SEARCH (Semantic + Keyword) (Средний приоритет)

### Описание:
Комбинировать семантический поиск с keyword matching для лучшего покрытия.

### Реализация:
```python
# Semantic search
semantic_results = semantic_search(task_prompt, files)
# Keyword search (BM25 или TF-IDF)
keyword_results = keyword_search(extract_keywords(task_prompt), files)
# Объединить результаты с весами
hybrid_results = combine_results(
    semantic_results, weight=0.7,
    keyword_results, weight=0.3
)
```

### Ожидаемый эффект: +0.05-0.08 балла
### Сложность: Средняя
### Приоритет: ⭐⭐⭐⭐

---

## 6. CONTEXT-AWARE CHUNKING (Средний приоритет)

### Описание:
Разбивать файлы на чанки с учетом структуры кода (классы, функции), а не просто по размеру.

### Реализация:
```python
def smart_chunk_by_structure(file_content, language):
    if language == 'java':
        # Разбить по классам/интерфейсам
        chunks = split_by_classes(file_content)
        # Каждый чанк = один класс + его методы
    elif language == 'python':
        chunks = split_by_functions_and_classes(file_content)
    return chunks
```

### Ожидаемый эффект: +0.03-0.06 балла
### Сложность: Средняя
### Приоритет: ⭐⭐⭐

---

## 7. MULTI-QUERY RETRIEVAL (Средний приоритет)

### Описание:
Генерировать несколько вариантов запроса и объединять результаты.

### Реализация:
```python
# Генерировать варианты запроса
queries = [
    task_prompt,  # Оригинальный
    generate_architectural_query(task_prompt),  # Архитектурный фокус
    generate_implementation_query(task_prompt),  # Фокус на реализацию
]
# Выполнить поиск для каждого запроса
results_per_query = [semantic_search(q, files) for q in queries]
# Объединить с reciprocal rank fusion
combined = reciprocal_rank_fusion(results_per_query)
```

### Ожидаемый эффект: +0.05-0.08 балла
### Сложность: Средняя
### Приоритет: ⭐⭐⭐⭐

---

## 8. RECIPROCAL RANK FUSION (RRF) (Низкий приоритет)

### Описание:
Объединять результаты из разных источников (semantic, keyword, graph) с RRF.

### Реализация:
```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            scores[item] = scores.get(item, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Ожидаемый эффект: +0.03-0.05 балла
### Сложность: Низкая
### Приоритет: ⭐⭐⭐

---

## 9. УЛУЧШЕННЫЙ АНАЛИЗ ЗАВИСИМОСТЕЙ (Высокий приоритет)

### Описание:
Более точный анализ зависимостей с учетом типов, наследования, интерфейсов.

### Реализация:
```python
# Для Java: парсить AST для точного определения зависимостей
import javalang
tree = javalang.parse.parse(file_content)
# Найти все imports, extends, implements
dependencies = extract_precise_dependencies(tree)
# Построить типизированный граф зависимостей
typed_graph = build_typed_dependency_graph(dependencies)
```

### Ожидаемый эффект: +0.05-0.10 балла
### Сложность: Высокая
### Приоритет: ⭐⭐⭐⭐

---

## 10. FILE IMPORTANCE SCORING (Средний приоритет)

### Описание:
Оценивать важность файлов не только по similarity, но и по структурной важности.

### Реализация:
```python
def calculate_file_importance(file_info, dependency_graph):
    score = 0.0
    # Количество файлов, зависящих от этого файла (reverse dependencies)
    dependents_count = len(get_reverse_dependencies(file_info, dependency_graph))
    score += dependents_count * 0.1
    
    # Центральность в графе (betweenness centrality)
    centrality = calculate_centrality(file_info, dependency_graph)
    score += centrality * 0.2
    
    # Размер файла (нормализованный)
    size_score = normalize_size(file_info['size'])
    score += size_score * 0.1
    
    return score
```

### Ожидаемый эффект: +0.03-0.06 балла
### Сложность: Средняя
### Приоритет: ⭐⭐⭐

---

## 11. ADAPTIVE RETRIEVAL ПО ТИПАМ ФАЙЛОВ (Средний приоритет)

### Описание:
Разные стратегии для разных типов файлов (тесты, конфиги, основной код).

### Реализация:
```python
def adaptive_retrieval_by_file_type(files, task_prompt):
    # Разделить файлы по типам
    test_files = [f for f in files if is_test_file(f)]
    config_files = [f for f in files if is_config_file(f)]
    code_files = [f for f in files if is_code_file(f)]
    
    # Разные стратегии для разных типов
    if is_testing_task(task_prompt):
        # Для тестовых задач: больше тестовых файлов
        selected = select_files(test_files, ratio=0.4) + select_files(code_files, ratio=0.6)
    else:
        # Для обычных задач: меньше тестовых файлов
        selected = select_files(code_files, ratio=0.9) + select_files(config_files, ratio=0.1)
    
    return selected
```

### Ожидаемый эффект: +0.03-0.05 балла
### Сложность: Низкая
### Приоритет: ⭐⭐⭐

---

## 12. CONTEXT COMPRESSION (Низкий приоритет)

### Описание:
Сжимать контекст, сохраняя важную информацию (удалять комментарии, форматирование).

### Реализация:
```python
def compress_code_context(file_content):
    # Удалить комментарии
    content = remove_comments(file_content)
    # Удалить лишние пробелы
    content = normalize_whitespace(content)
    # Сохранить только сигнатуры методов для больших файлов
    if len(content) > 5000:
        content = extract_method_signatures(content) + extract_class_definitions(content)
    return content
```

### Ожидаемый эффект: +0.02-0.04 балла (косвенно - больше файлов поместится)
### Сложность: Низкая
### Приоритет: ⭐⭐

---

## 13. BETTER PROMPT ENGINEERING ДЛЯ RETRIEVAL (Средний приоритет)

### Описание:
Создавать специальные промпты для ритривера, извлекая ключевую информацию из задачи.

### Реализация:
```python
def create_retrieval_query(task_prompt):
    # Извлечь ключевые сущности
    entities = extract_entities(task_prompt)  # RoomStore, OfflineSyncWorker
    # Извлечь действия
    actions = extract_actions(task_prompt)  # merge, refactor, implement
    # Извлечь доменные концепции
    concepts = extract_concepts(task_prompt)  # sync, offline, persistence
    
    # Создать оптимизированный запрос для ритривера
    retrieval_query = f"{' '.join(entities)} {' '.join(actions)} {' '.join(concepts)}"
    return retrieval_query
```

### Ожидаемый эффект: +0.05-0.08 балла
### Сложность: Низкая
### Приоритет: ⭐⭐⭐⭐

---

## 14. FINE-TUNING EMBEDDING MODEL (Низкий приоритет - долго)

### Описание:
Fine-tune модель эмбеддингов на задачах из LoCoBench для лучшего понимания кода.

### Реализация:
```python
# Использовать SentenceTransformers для fine-tuning
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer('all-MiniLM-L6-v2')
# Подготовить данные из LoCoBench
train_examples = prepare_training_data_from_locobench()
# Fine-tune
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)
```

### Ожидаемый эффект: +0.10-0.20 балла (но требует времени на обучение)
### Сложность: Высокая
### Приоритет: ⭐⭐

---

## 15. ИСПОЛЬЗОВАНИЕ БОЛЬШИХ/ЛУЧШИХ МОДЕЛЕЙ (Средний приоритет)

### Описание:
Использовать более мощные модели эмбеддингов (например, code-specific модели).

### Реализация:
```python
# Использовать модели, специально обученные на коде
better_models = [
    'microsoft/codebert-base',  # CodeBERT
    'Salesforce/codet5-base',   # CodeT5
    'sentence-transformers/all-mpnet-base-v2',  # Более мощная общая модель
]

# Или использовать локальную модель VESO (уже настроена)
model = SentenceTransformer('/srv/nfs/VESO/models/veso-models/VESO-30alpha3')
```

### Ожидаемый эффект: +0.05-0.15 балла
### Сложность: Низкая (если модель доступна)
### Приоритет: ⭐⭐⭐⭐

---

## 16. ITERATIVE RETRIEVAL (Средний приоритет)

### Описание:
Итеративно улучшать результаты: сначала найти базовые файлы, потом найти связанные.

### Реализация:
```python
def iterative_retrieval(task_prompt, files, iterations=2):
    selected_files = []
    
    # Итерация 1: Найти семантически релевантные файлы
    semantic_files = semantic_search(task_prompt, files, top_k=20)
    selected_files.extend(semantic_files)
    
    # Итерация 2: Найти файлы, связанные с найденными
    for iteration in range(iterations - 1):
        related_files = find_related_files(selected_files, files)
        # Ранжировать связанные файлы по релевантности к задаче
        ranked_related = rank_files(related_files, task_prompt)
        selected_files.extend(ranked_related[:10])
    
    return deduplicate(selected_files)
```

### Ожидаемый эффект: +0.05-0.08 балла
### Сложность: Средняя
### Приоритет: ⭐⭐⭐

---

## 17. TASK-SPECIFIC RETRIEVAL STRATEGIES (Высокий приоритет)

### Описание:
Специализированные стратегии для разных типов задач (уже частично реализовано, можно улучшить).

### Реализация:
```python
# Для архитектурных задач: больше внимания к интерфейсам и абстракциям
if is_architectural_task:
    # Увеличить вес для файлов с интерфейсами
    boost_interface_files = True
    # Больше зависимостей
    dependency_depth = 3  # вместо 2
    # Приоритет файлам с паттернами проектирования
    boost_design_patterns = True

# Для code comprehension: больше внимания к потоку данных
elif is_code_comprehension_task:
    # Следовать по цепочке вызовов
    follow_call_chain = True
    # Включать все файлы в цепочке
    include_full_chain = True
```

### Ожидаемый эффект: +0.05-0.10 балла
### Сложность: Средняя
### Приоритет: ⭐⭐⭐⭐⭐

---

## 18. SEMANTIC CLUSTERING (Низкий приоритет)

### Описание:
Кластеризовать файлы по семантической близости и выбирать представителей из каждого кластера.

### Реализация:
```python
from sklearn.cluster import KMeans

# Кластеризовать файлы по эмбеддингам
embeddings = model.encode([f['content'] for f in files])
clusters = KMeans(n_clusters=10).fit(embeddings)

# Выбрать лучший файл из каждого кластера
selected = []
for cluster_id in range(10):
    cluster_files = [files[i] for i, c in enumerate(clusters.labels_) if c == cluster_id]
    best_in_cluster = rank_files(cluster_files, task_prompt)[0]
    selected.append(best_in_cluster)
```

### Ожидаемый эффект: +0.03-0.05 балла
### Сложность: Средняя
### Приоритет: ⭐⭐⭐

---

## 19. ACTIVE LEARNING / RELEVANCE FEEDBACK (Низкий приоритет)

### Описание:
Использовать обратную связь от модели для улучшения ритривера (сложно в текущей архитектуре).

### Ожидаемый эффект: +0.05-0.10 балла
### Сложность: Высокая
### Приоритет: ⭐⭐

---

## 20. MULTI-MODAL RETRIEVAL (Низкий приоритет)

### Описание:
Использовать не только текст, но и структуру (AST, граф зависимостей) для ритривера.

### Реализация:
```python
# Эмбеддинги для структуры кода
ast_embeddings = encode_ast_structure(files)
# Эмбеддинги для текста
text_embeddings = encode_text(files)
# Объединить
combined_embeddings = combine_embeddings(ast_embeddings, text_embeddings)
```

### Ожидаемый эффект: +0.05-0.10 балла
### Сложность: Высокая
### Приоритет: ⭐⭐

---

## ПРИОРИТЕТНЫЙ ПЛАН РЕАЛИЗАЦИИ

### Фаза 1: Быстрые улучшения (1-2 дня)
1. ✅ Query Expansion (+0.05-0.10)
2. ✅ Better Prompt Engineering для Retrieval (+0.05-0.08)
3. ✅ Улучшенный анализ зависимостей (+0.05-0.10)
4. ✅ Task-Specific Strategies (улучшить существующие) (+0.05-0.10)

**Ожидаемый эффект Фазы 1: +0.20-0.38 балла → Total Score: 2.30-2.50**

### Фаза 2: Средние улучшения (3-5 дней)
5. ✅ Re-ranking с Cross-Encoder (+0.10-0.15)
6. ✅ Graph-Based Retrieval (расширить существующий) (+0.08-0.12)
7. ✅ Multi-Query Retrieval (+0.05-0.08)
8. ✅ Hybrid Search (+0.05-0.08)

**Ожидаемый эффект Фазы 2: +0.28-0.43 балла → Total Score: 2.40-2.65**

### Фаза 3: Продвинутые улучшения (по необходимости)
9. ✅ Hierarchical Retrieval (+0.05-0.08)
10. ✅ Context-Aware Chunking (+0.03-0.06)
11. ✅ File Importance Scoring (+0.03-0.06)

**Ожидаемый эффект Фазы 3: +0.11-0.20 балла → Total Score: 2.50-2.85**

---

## РЕКОМЕНДАЦИИ ДЛЯ БЫСТРОГО РЕЗУЛЬТАТА

**Начать с:**
1. Query Expansion - быстро реализовать, хороший эффект
2. Better Prompt Engineering - легко добавить
3. Re-ranking с Cross-Encoder - требует установки библиотеки, но дает большой эффект
4. Улучшить Graph-Based Retrieval - расширить существующий код

**Эти 4 метода должны дать +0.25-0.45 балла, что достаточно для достижения 2.3+**
