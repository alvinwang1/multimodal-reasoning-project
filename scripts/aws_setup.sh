#!/usr/bin/env bash
# One-time setup of an AWS GPU instance for the final-report runs (README, "Final report runs on AWS").
# Target: g6.2xlarge (1x NVIDIA L4, 8 vCPU, 32 GiB), launched from an AWS Deep Learning AMI that includes the NVIDIA
# driver, with >= 100 GB of disk. From the repo root:
#
#   bash scripts/aws_setup.sh
#
# then:  tmux new -s run   ->   source .venv/bin/activate && python -m src.final.run_all
# Safe to rerun: finished steps are skipped (a failed package install is retried).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f requirements.txt ] || { echo ">> run this from inside the repo"; exit 1; }

echo ">> GPU"
command -v nvidia-smi >/dev/null || {
    echo ">> nvidia-smi not found: launch the instance from an AMI with the NVIDIA driver (an AWS Deep Learning AMI)"
    exit 1
}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo ">> tools (tmux, git, uv)"
for tool in tmux git; do
    command -v "$tool" >/dev/null && continue
    if command -v apt-get >/dev/null; then sudo apt-get update -qq && sudo apt-get install -y -qq "$tool"
    else sudo dnf install -y -q "$tool"; fi
done
if ! command -v uv >/dev/null; then              # uv provides Python 3.12 whatever the AMI ships
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ">> Python 3.12 environment in .venv with the team's pinned packages (requirements.txt)"
[ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv
if [ ! -f .venv/.requirements_ok ]; then        # marker: a failed install is retried on the next run
    # the annotation tool (gradio) and Empath are only for Assignment 2 analyses B and C
    grep -vE '^(gradio|empath)[=<> ]' requirements.txt > .venv/requirements_runs.txt
    uv pip install -q --python .venv/bin/python -r .venv/requirements_runs.txt
    touch .venv/.requirements_ok
fi

echo ">> PyTorch on the GPU"
.venv/bin/python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
assert torch.cuda.is_available(), ("PyTorch cannot use the GPU: the AMI's NVIDIA driver is probably older than this "
                                   "torch build needs. Use a current AWS Deep Learning AMI.")
torch.ones(8, device="cuda").sum().item()  # fails if this torch build has no kernels for this GPU
print("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
EOF

echo ">> downloading Qwen2.5-Omni-3B, Whisper, AST and OmniBench (about 15 GB, once)"
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
from src.common import load_config
from src.data import OmniBenchSubset
from src.final.idea4_generate import AST_MODEL, ASR_MODEL
cfg = load_config()
for repo in (cfg["model"]["hf_id"], ASR_MODEL, AST_MODEL):
    snapshot_download(repo)
data = OmniBenchSubset(cfg)
ex = data[data.ids[0]]  # checks that MP3 decoding works on this machine
print("dataset ok:", len(data.records), "usable items; first clip", round(len(ex["audio"]) / 16000, 1), "s")
EOF

echo ">> setup done (commit $(git rev-parse --short HEAD)). Next:"
echo "     tmux new -s run"
echo "     source .venv/bin/activate && python -m src.final.run_all"
