cd /root/autodl-tmp/visual-search-agent/train/OpenSearch-VL-Training

python -m pip install \
  setuptools==79.0.1 \
  wheel==0.45.1 \
  pybind11==2.13.6 \
  packaging==24.2 \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple

python -m pip install \
  -e ./RL/rllm \
  -e ./RL/rllm/verl \
  -e ./RL/Megatron-LM \
  -e ./RL/mbridge \
  "torch==2.8.0+cu128" \
  "torchvision==0.23.0+cu128" \
  "torchaudio==2.8.0+cu128" \
  "numpy==1.26.4" \
  "sglang==0.5.5" \
  "transformers==4.57.1" \
  "ray[default]==2.41.0" \
  "requests==2.32.3" \
  "modelscope==1.31.0" \
  python-dotenv qwen-vl-utils \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/ \
  --report .codex/install-training-dependencies.json