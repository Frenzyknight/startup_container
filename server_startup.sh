#!/bin/bash

pip install vllm==0.9.1
cd /workspace
git clone https://github.com/rednote-hilab/dots.ocr.git dots-ocr
cd dots-ocr
pip install --no-cache-dir -r requirements.txt
pip install -e .
python3 tools/download_model.py

export HF_MODEL_PATH="/workspace/dots-ocr/weights/DotsOCR"
export PYTHONPATH="/workspace/dots-ocr/weights:$PYTHONPATH"
sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\from DotsOCR import modeling_dots_ocr_vllm' $(which vllm)

vllm serve /workspace/dots-ocr/weights/DotsOCR \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --chat-template-content-format string \
    --served-model-name "dotsocr-model" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 40000
