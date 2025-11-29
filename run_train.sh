#!/bin/bash

# Dừng script ngay nếu có lệnh bị lỗi
set -e

echo "============================================"
echo "🚀 STARTING TRAINING PIPELINE FOR VIQUAD 2.0"
echo "============================================"

# 1. Cài đặt thư viện (Chỉ chạy nếu chưa cài, nhưng trên VastAI chạy lại cho chắc)
echo "[1/3] Installing requirements..."
pip install -r requirements.txt
pip install flash-attn --no-build-isolation # Cài riêng để tránh lỗi

# 2. Chạy Training
echo "--------------------------------------------"
echo "[2/3] Starting Training (Trainer)..."
echo "--------------------------------------------"
# python -m src.trainer  <-- Lệnh gốc
# Thêm accelerate launch để tối ưu GPU nếu muốn
python -m src.trainer

# 3. Chạy Evaluation (Optional)
echo "--------------------------------------------"
echo "[3/3] Running Evaluation & Generation..."
echo "--------------------------------------------"
python -m src.eval_generation

echo "============================================"
echo "✅ PIPELINE COMPLETED SUCCESSFULLY!"
echo "Check outputs in ./checkpoints/qwen_viquad_final"
echo "============================================"