#!/bin/bash
# Start the deepfake detection consumer service

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# ENSEMBLE CONFIGURATION (3x inference time, 3x memory)
# =============================================================================
export USE_ENSEMBLE="${USE_ENSEMBLE:-1}"
export ENSEMBLE_METHOD="${ENSEMBLE_METHOD:-weighted_avg}"

# Model paths (ensemble mode)
export TCN_MODEL_PATH="${TCN_MODEL_PATH:-$SCRIPT_DIR/TCN_TemporalConvNet_Residual Blocks_final.h5}"
export BILSTM_MODEL_PATH="${BILSTM_MODEL_PATH:-$SCRIPT_DIR/BiLSTM_Multi_Head_Attention_HMM_final.h5}"
export BIGRU_MODEL_PATH="${BIGRU_MODEL_PATH:-$SCRIPT_DIR/BiGRU_Multi_Head_Attention_HMM_final.h5}"

# Single model path (fallback if USE_ENSEMBLE=0)
export MODEL_PATH="${MODEL_PATH:-$SCRIPT_DIR/TCN_TemporalConvNet_Residual Blocks_final.h5}"

# Scaler path
export SCALER_PATH="${SCALER_PATH:-$SCRIPT_DIR/csv_scaler.pkl}"

# Ensemble weights - 2 models only (BiGRU has Keras compatibility issues)
# TCN=0.841acc (weight 0.504), BiLSTM=0.827acc (weight 0.496)
export TCN_WEIGHT="${TCN_WEIGHT:-0.504}"
export BILSTM_WEIGHT="${BILSTM_WEIGHT:-0.496}"
export BIGRU_WEIGHT="${BIGRU_WEIGHT:-0.0}"

# =============================================================================
# RABBITMQ & PROCESSING CONFIGURATION
# =============================================================================
export RABBITMQ_URL="${RABBITMQ_URL:-amqp://admin:P@ssw0rd123@localhost:5672/}"
export RABBITMQ_QUEUE="${RABBITMQ_QUEUE:-feature.extraction.results}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export PAD_SEQUENCE="${PAD_SEQUENCE:-1}"
export REQUEUE_ON_ERROR="${REQUEUE_ON_ERROR:-0}"
export DELETE_FRAMES="${DELETE_FRAMES:-0}"

# =============================================================================
# SOCKETIO CONFIGURATION
# =============================================================================
export SOCKETIO_ENABLED="${SOCKETIO_ENABLED:-1}"
export SOCKETIO_HOST="${SOCKETIO_HOST:-0.0.0.0}"
export SOCKETIO_PORT="${SOCKETIO_PORT:-8093}"
export SOCKETIO_USE_ROOMS="${SOCKETIO_USE_ROOMS:-1}"

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================
echo "=========================================="
echo "Deepfake Detection Consumer"
echo "=========================================="
echo "RabbitMQ: $RABBITMQ_URL"
echo "Queue: $RABBITMQ_QUEUE"
echo "Log Level: $LOG_LEVEL"
echo "Padding: $PAD_SEQUENCE"
echo ""

if [ "$USE_ENSEMBLE" = "1" ]; then
    if [ "$BIGRU_WEIGHT" = "0.0" ] || [ "$BIGRU_WEIGHT" = "0" ]; then
        echo "🤖 MODE: Ensemble (2 models - TCN + BiLSTM)"
        echo "   Method: $ENSEMBLE_METHOD"
        echo "   Weights: TCN=$TCN_WEIGHT BiLSTM=$BILSTM_WEIGHT"
        echo "   Note: BiGRU disabled (Keras compatibility issues)"
    else
        echo "🤖 MODE: Ensemble (3 models)"
        echo "   Method: $ENSEMBLE_METHOD"
        echo "   Weights: TCN=$TCN_WEIGHT BiLSTM=$BILSTM_WEIGHT BiGRU=$BIGRU_WEIGHT"
    fi
    echo "   ⚡ Parallel Inference: ENABLED (optimized)"
    echo "   ⚡ Inference time: ~60-80ms (was ~120-180ms sequential)"
    echo "   ⚠️  Memory usage: ~2-3x higher"
    echo ""
    echo "Models:"
    [ -f "$TCN_MODEL_PATH" ] && echo "   ✅ TCN" || echo "   ❌ TCN (not found)"
    [ -f "$BILSTM_MODEL_PATH" ] && echo "   ✅ BiLSTM" || echo "   ❌ BiLSTM (not found)"
    if [ "$BIGRU_WEIGHT" != "0.0" ] && [ "$BIGRU_WEIGHT" != "0" ]; then
        [ -f "$BIGRU_MODEL_PATH" ] && echo "   ✅ BiGRU" || echo "   ❌ BiGRU (not found)"
    fi
else
    echo "🤖 MODE: Single Model"
    echo "   Path: $MODEL_PATH"
    echo "   ⚡ Inference time: ~40-60ms"
    [ -f "$MODEL_PATH" ] && echo "   ✅ Model ready" || echo "   ❌ Model not found"
fi

echo ""
echo "Socket.IO: $SOCKETIO_HOST:$SOCKETIO_PORT (enabled=$SOCKETIO_ENABLED)"
echo "=========================================="
echo ""

# Run consumer (will block and continuously consume messages)
python3 consumer.py
