#!/bin/bash
# Idea 3 on PSC Bridges-2: run the budget grid on a GPU, analyze, commit + push the results.
#
#   bash scripts/idea3_psc.sh setup   # once, on a LOGIN node: env + model/dataset download
#   bash scripts/idea3_psc.sh         # on a GPU node (interactive or sbatch): run, analyze, commit, push
#
# Interactive:  interact -p GPU-shared --gres=gpu:h100-80:1 -t 04:00:00
# Batch:        sbatch scripts/idea3_psc.sh
# V100:         sbatch --gpus=v100-32:1 -t 08:00:00 scripts/idea3_psc.sh   (interact: --gres=gpu:v100-32:1)
#   The default PyTorch wheel has no V100 (sm_70) kernels, so the script swaps in the cu126 build of the
#   same version. V100 has no bf16 and fp16 can overflow, so it runs in fp32 (~3x slower than H100 bf16).
#   Override with DTYPE=float16 bash scripts/idea3_psc.sh.
# Resumable: if the job dies, run it again and it continues where it stopped.
#
#SBATCH -J idea3
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -t 04:00:00
#SBATCH -o idea3_%j.out
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

STORE="${PROJECT:-$HOME}"                       # $PROJECT has the quota; $HOME on Bridges-2 is small
ENV_DIR="$STORE/envs/mmr"
export HF_HOME="$STORE/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

activate() {
    module load anaconda3 2>/dev/null || true
    if [ ! -x "$ENV_DIR/bin/python" ]; then
        echo ">> creating env at $ENV_DIR"
        conda create -y -q -p "$ENV_DIR" python=3.12
        "$ENV_DIR/bin/pip" install -q -r requirements.txt
    fi
    export PATH="$ENV_DIR/bin:$PATH"
}

download() {
    echo ">> fetching model + dataset into $HF_HOME"
    python - <<'EOF'
from huggingface_hub import snapshot_download
from src.common import load_config
from src.data import OmniBenchSubset
cfg = load_config()
snapshot_download(cfg["model"]["hf_id"])
data = OmniBenchSubset(cfg)
ex = data[data.ids[0]]  # checks MP3 decoding works on this machine
print("dataset ok; audio samples:", len(ex["audio"]))
EOF
}

activate
if [ "${1:-}" = "setup" ]; then
    download
    echo ">> setup done. On a GPU node run: bash scripts/idea3_psc.sh"
    exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# is this GPU's architecture compiled into the installed torch? (no on V100 with the default wheel)
has_arch() {
    python -c "import torch; c = torch.cuda.get_device_capability(); \
assert f'sm_{c[0]}{c[1]}' in torch.cuda.get_arch_list(); torch.ones(8, device='cuda').sum().item()" 2>/dev/null
}
if ! has_arch; then
    TV=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    VV=$(python -c "import torchvision; print(torchvision.__version__.split('+')[0])")
    echo ">> torch $TV lacks kernels for this GPU; installing the cu126 build"
    pip install -q "torch==$TV+cu126" "torchvision==$VV+cu126" \
        --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
    has_arch || { echo ">> still no kernels for this GPU; request an H100 (h100-80)"; exit 1; }
fi
# bf16 needs compute capability >= 8 (A100/H100); older GPUs (V100) run fp32
DTYPE=${DTYPE:-$(python -c "import torch; print('bfloat16' if torch.cuda.get_device_capability()[0] >= 8 else 'float32')")}
echo ">> dtype $DTYPE"

# compute nodes may lack internet: use the cache from `setup` when it is there
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Omni-3B', local_files_only=True)" \
    2>/dev/null && export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 || echo ">> no local model cache; downloading"

python -m src.idea3.run --split dev --device cuda --dtype "$DTYPE" --limit 1   # smoke test (resumable, kept)
python -m src.idea3.run --split dev --device cuda --dtype "$DTYPE"
python -m src.idea3.run --split test --device cuda --dtype "$DTYPE"
python -m src.idea3.analyze

echo ">> committing results"
git add results/idea3 data/idea3_test_ids.json
if git diff --cached --quiet; then
    echo ">> nothing new to commit"
else
    git commit -q -m "Idea 3: budget-grid results ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), $DTYPE)"
    git pull -q --rebase && git push -q && echo ">> pushed" \
        || echo ">> commit is local; push failed (check GitHub SSH key on PSC), then: git pull --rebase && git push"
fi
