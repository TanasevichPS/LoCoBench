# Исправление разрешения путей для MCP

## ✅ Проблема исправлена

### Проблема
Пути в `context_files` имеют формат `"EduGate_ScholarLink//src//components//validator.c"`, а `project_dir` уже указывает на `.../EduGate_ScholarLink`. При попытке загрузить файл получается неправильный путь.

### Решение

1. **Обновлена функция `_normalize_relative_path()`**:
   - Теперь нормализует двойные слеши `//` в одинарные `/`

2. **Обновлена функция `load_context_files_from_scenario()`**:
   - Определяет базовое имя проекта из `project_dir`
   - Удаляет префикс проекта из путей в `context_files`
   - Пробует оба варианта пути (с префиксом и без)

3. **Улучшено логирование**:
   - Более подробные сообщения об ошибках загрузки файлов

## 🔧 Как это работает

### Пример:

**Сценарий:**
- `project_dir` = `/path/to/data/generated/c_api_gateway_easy_009/EduGate_ScholarLink`
- `context_files` = `["EduGate_ScholarLink//src//components//validator.c"]`

**Процесс:**
1. Нормализация: `"EduGate_ScholarLink//src//components//validator.c"` → `"EduGate_ScholarLink/src/components/validator.c"`
2. Определение базового имени: `project_dir.name` = `"EduGate_ScholarLink"`
3. Удаление префикса: `"EduGate_ScholarLink/src/components/validator.c"` → `"src/components/validator.c"`
4. Формирование пути: `project_dir / "src/components/validator.c"` = `/path/to/.../EduGate_ScholarLink/src/components/validator.c`
5. Загрузка файла

## ✅ Теперь должно работать

MCP tools теперь должны правильно загружать файлы по путям из `context_files`.

Запустите снова:

```bash
python -m locobench.cli evaluate \
    --model "DeepSeekR1-70B-LRI" \
    --config-path config.yaml
```

В логах вы должны увидеть:
```
INFO: 📋 Loaded X files from scenario['context_files'] list for retrieval
INFO: 📁 Loaded X files from project_dir for MCP tools
INFO: ✅ Tool 'find_security_sensitive_files': found X files
```
