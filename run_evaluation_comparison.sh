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

# Создаем директорию для чекпоинтов с ритривером
mkdir -p "data/output_with_retrieval_${TIMESTAMP}/intermediate_results"

OUTPUT_FILE_WITH="$RESULTS_WITH_RETRIEVAL/evaluation_results.json"

echo "🚀 Starting evaluation with retrieval..."
echo "   Config: $CONFIG_WITH_RETRIEVAL"
echo "   Output: $OUTPUT_FILE_WITH"
echo "   Checkpoints: data/output_with_retrieval_${TIMESTAMP}/intermediate_results/"
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

# Создаем директорию для чекпоинтов без ритривера
mkdir -p "data/output_without_retrieval_${TIMESTAMP}/intermediate_results"

OUTPUT_FILE_WITHOUT="$RESULTS_WITHOUT_RETRIEVAL/evaluation_results.json"

echo "🚀 Starting evaluation without retrieval..."
echo "   Config: $CONFIG_WITHOUT_RETRIEVAL"
echo "   Output: $OUTPUT_FILE_WITHOUT"
echo "   Checkpoints: data/output_without_retrieval_${TIMESTAMP}/intermediate_results/"
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
