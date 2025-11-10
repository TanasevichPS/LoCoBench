#!/bin/bash

# Скрипт для последовательного запуска оценок: сначала с ритривером, потом без
# Результаты и чекпоинты сохраняются в разных папках для изоляции

set -e  # Остановка при ошибке

MODEL="${1:-DeepSeekR1-70B-LRI}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "LoCoBench Evaluation Comparison"
echo "=========================================="
echo "Model: $MODEL"
echo "Timestamp: $TIMESTAMP"
echo ""

# Создаем директории для результатов
RESULTS_WITH_RETRIEVAL="evaluation_results/with_retrieval_${TIMESTAMP}"
RESULTS_WITHOUT_RETRIEVAL="evaluation_results/without_retrieval_${TIMESTAMP}"
CHECKPOINTS_WITH_RETRIEVAL="data/output/intermediate_results/with_retrieval_${TIMESTAMP}"
CHECKPOINTS_WITHOUT_RETRIEVAL="data/output/intermediate_results/without_retrieval_${TIMESTAMP}"

mkdir -p "$RESULTS_WITH_RETRIEVAL"
mkdir -p "$RESULTS_WITHOUT_RETRIEVAL"
mkdir -p "$CHECKPOINTS_WITH_RETRIEVAL"
mkdir -p "$CHECKPOINTS_WITHOUT_RETRIEVAL"

# Создаем временные конфигурации
CONFIG_WITH_RETRIEVAL="config_with_retrieval_${TIMESTAMP}.yaml"
CONFIG_WITHOUT_RETRIEVAL="config_without_retrieval_${TIMESTAMP}.yaml"

# Копируем базовый конфиг и настраиваем для ритривера
cp config.yaml "$CONFIG_WITH_RETRIEVAL"
# Убеждаемся что ритривер включен (уже включен по умолчанию)
sed -i 's/enabled: false/enabled: true/' "$CONFIG_WITH_RETRIEVAL" || true

# Копируем базовый конфиг и отключаем ритривер
cp config.yaml "$CONFIG_WITHOUT_RETRIEVAL"
sed -i 's/enabled: true/enabled: false/' "$CONFIG_WITHOUT_RETRIEVAL"

# Изменяем output_dir в конфигах для изоляции чекпоинтов
# Для конфига с ритривером
sed -i "s|output_dir: \"./data/output\"|output_dir: \"./data/output_with_retrieval_${TIMESTAMP}\"|" "$CONFIG_WITH_RETRIEVAL"
# Для конфига без ритривера
sed -i "s|output_dir: \"./data/output\"|output_dir: \"./data/output_without_retrieval_${TIMESTAMP}\"|" "$CONFIG_WITHOUT_RETRIEVAL"

echo "📁 Created temporary configs:"
echo "   - $CONFIG_WITH_RETRIEVAL (retrieval enabled)"
echo "   - $CONFIG_WITHOUT_RETRIEVAL (retrieval disabled)"
echo ""

# ==========================================
# ШАГ 1: Запуск с ритривером
# ==========================================
echo "=========================================="
echo "STEP 1: Evaluation WITH Retrieval"
echo "=========================================="
echo ""

# Определяем директорию для чекпоинтов из конфига
OUTPUT_DIR_WITH=$(grep "output_dir:" "$CONFIG_WITH_RETRIEVAL" | awk '{print $2}' | tr -d '"')
INTERMEDIATE_DIR_WITH="${OUTPUT_DIR_WITH}/intermediate_results"

# Очищаем чекпоинты перед запуском (все возможные пути)
echo "🧹 Cleaning up checkpoints before evaluation..."
CLEANED=0

# 1. Очищаем из конфига
if [ -d "$INTERMEDIATE_DIR_WITH" ]; then
    echo "   Removing (from config): $INTERMEDIATE_DIR_WITH"
    rm -rf "$INTERMEDIATE_DIR_WITH"
    CLEANED=1
fi

# 2. Очищаем абсолютный путь (может быть жестко закодирован)
ABSOLUTE_INTERMEDIATE="/srv/nfs/VESO/home/polina/trsh/LoCoBench/intermediate_results"
if [ -d "$ABSOLUTE_INTERMEDIATE" ]; then
    echo "   Removing (absolute path): $ABSOLUTE_INTERMEDIATE"
    rm -rf "$ABSOLUTE_INTERMEDIATE"
    CLEANED=1
fi

# 3. Очищаем другие возможные пути
if [ -d "./intermediate_results" ]; then
    echo "   Removing: ./intermediate_results"
    rm -rf "./intermediate_results"
    CLEANED=1
fi

if [ -d "data/output/intermediate_results" ]; then
    echo "   Removing: data/output/intermediate_results"
    rm -rf "data/output/intermediate_results"
    CLEANED=1
fi

if [ $CLEANED -eq 0 ]; then
    echo "   ℹ️  No intermediate_results directories found to clean"
else
    echo "   ✅ Cleaned up all intermediate_results directories"
fi
echo ""

OUTPUT_FILE_WITH="$RESULTS_WITH_RETRIEVAL/evaluation_results.json"

echo "🚀 Starting evaluation with retrieval..."
echo "   Config: $CONFIG_WITH_RETRIEVAL"
echo "   Output: $OUTPUT_FILE_WITH"
echo "   Checkpoints: $INTERMEDIATE_DIR_WITH"
echo ""

locobench evaluate \
    --config-path "$CONFIG_WITH_RETRIEVAL" \
    --model "$MODEL" \
    --output-file "$OUTPUT_FILE_WITH" \
    --no-resume

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation WITH retrieval completed successfully!"
    echo "   Results saved to: $OUTPUT_FILE_WITH"
else
    echo ""
    echo "❌ Evaluation WITH retrieval failed!"
    exit 1
fi

echo ""
echo "Waiting 5 seconds before next evaluation..."
sleep 5

# ==========================================
# ШАГ 2: Запуск без ритривера
# ==========================================
echo ""
echo "=========================================="
echo "STEP 2: Evaluation WITHOUT Retrieval"
echo "=========================================="
echo ""

# Определяем директорию для чекпоинтов из конфига
OUTPUT_DIR_WITHOUT=$(grep "output_dir:" "$CONFIG_WITHOUT_RETRIEVAL" | awk '{print $2}' | tr -d '"')
INTERMEDIATE_DIR_WITHOUT="${OUTPUT_DIR_WITHOUT}/intermediate_results"

# Очищаем чекпоинты перед запуском (все возможные пути)
echo "🧹 Cleaning up checkpoints before evaluation..."
CLEANED=0

# 1. Очищаем из конфига
if [ -d "$INTERMEDIATE_DIR_WITHOUT" ]; then
    echo "   Removing (from config): $INTERMEDIATE_DIR_WITHOUT"
    rm -rf "$INTERMEDIATE_DIR_WITHOUT"
    CLEANED=1
fi

# 2. Очищаем абсолютный путь (может быть жестко закодирован)
ABSOLUTE_INTERMEDIATE="/srv/nfs/VESO/home/polina/trsh/LoCoBench/intermediate_results"
if [ -d "$ABSOLUTE_INTERMEDIATE" ]; then
    echo "   Removing (absolute path): $ABSOLUTE_INTERMEDIATE"
    rm -rf "$ABSOLUTE_INTERMEDIATE"
    CLEANED=1
fi

# 3. Очищаем другие возможные пути
if [ -d "./intermediate_results" ]; then
    echo "   Removing: ./intermediate_results"
    rm -rf "./intermediate_results"
    CLEANED=1
fi

if [ -d "data/output/intermediate_results" ]; then
    echo "   Removing: data/output/intermediate_results"
    rm -rf "data/output/intermediate_results"
    CLEANED=1
fi

if [ $CLEANED -eq 0 ]; then
    echo "   ℹ️  No intermediate_results directories found to clean"
else
    echo "   ✅ Cleaned up all intermediate_results directories"
fi
echo ""

OUTPUT_FILE_WITHOUT="$RESULTS_WITHOUT_RETRIEVAL/evaluation_results.json"

echo "🚀 Starting evaluation without retrieval..."
echo "   Config: $CONFIG_WITHOUT_RETRIEVAL"
echo "   Output: $OUTPUT_FILE_WITHOUT"
echo "   Checkpoints: $INTERMEDIATE_DIR_WITHOUT"
echo ""

locobench evaluate \
    --config-path "$CONFIG_WITHOUT_RETRIEVAL" \
    --model "$MODEL" \
    --output-file "$OUTPUT_FILE_WITHOUT" \
    --no-resume

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation WITHOUT retrieval completed successfully!"
    echo "   Results saved to: $OUTPUT_FILE_WITHOUT"
else
    echo ""
    echo "❌ Evaluation WITHOUT retrieval failed!"
    exit 1
fi

# ==========================================
# Итоги
# ==========================================
echo ""
echo "=========================================="
echo "Evaluation Comparison Complete!"
echo "=========================================="
echo ""
echo "📊 Results:"
echo "   WITH retrieval:    $OUTPUT_FILE_WITH"
echo "   WITHOUT retrieval:  $OUTPUT_FILE_WITHOUT"
echo ""
echo "💾 Checkpoints (isolated):"
echo "   WITH retrieval:    data/output_with_retrieval_${TIMESTAMP}/intermediate_results/"
echo "   WITHOUT retrieval:  data/output_without_retrieval_${TIMESTAMP}/intermediate_results/"
echo ""
echo "🧹 Temporary configs (can be deleted):"
echo "   $CONFIG_WITH_RETRIEVAL"
echo "   $CONFIG_WITHOUT_RETRIEVAL"
echo ""
