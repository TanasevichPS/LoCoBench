# Проверка исправления MCP параметров

## ✅ Параметры добавлены

Проверьте, что в файле `locobench/retrieval.py` на строках **1298-1303** есть:

```python
    use_mcp: bool = False,  # Use MCP-based retrieval
    mcp_provider: Optional[str] = None,  # MCP provider
    mcp_model: Optional[str] = None,  # MCP model
    mcp_base_url: Optional[str] = None,  # MCP base URL
    mcp_api_key: Optional[str] = None,  # MCP API key
    config: Optional[Any] = None,  # Config object
```

## 🔍 Проверка

Выполните:

```bash
# Проверить наличие параметров
grep -n "use_mcp: bool" locobench/retrieval.py
grep -n "mcp_provider:" locobench/retrieval.py
```

Должны увидеть строки 1298 и 1299.

## 🧹 Очистка кэша

```bash
# Очистить кэш Python
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Перезапустить Python процесс (если запущен)
```

## ✅ Все параметры имеют дефолтные значения

Все параметры MCP имеют дефолтные значения:
- `use_mcp: bool = False` - по умолчанию отключено
- `mcp_provider: Optional[str] = None` - опционально
- И т.д.

Это означает, что функция должна работать даже если эти параметры не переданы.

## 🚀 Тест

После очистки кэша попробуйте снова:

```bash
python -m locobench.cli evaluate \
    --scenarios data/output/scenarios/test_easy_scenario.json \
    --config config.yaml
```

Если ошибка сохраняется, проверьте:
1. Что файл `locobench/retrieval.py` действительно обновлен
2. Что нет других версий файла
3. Что Python процесс полностью перезапущен
