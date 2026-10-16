#!/bin/bash
#SBATCH --job-name=mvlm-download
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/%x-%j.out
#
# Download model weights into $HF_HOME on a CPU node. Submit from the
# repository root with the model configs to fetch (SAM 2 and SAM 3 are always
# included; SAM 3 is skipped with a hint if access has not been granted):
#   sbatch scripts/euler/download.sbatch qwen2.5-vl-72b qwen3-vl-32b qwen3.5-27b
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/euler/env.sh
source "$VENV_DIR/bin/activate"

ids=$(python - "$@" <<'EOF'
import sys, yaml
default = yaml.safe_load(open("configs/default.yaml"))
ids = [default["segmenter_id"], "facebook/sam3"]
for name in sys.argv[1:]:
    ids.append(yaml.safe_load(open(f"configs/models/{name}.yaml"))["model_id"])
print(" ".join(ids))
EOF
)

for id in $ids; do
    echo "Downloading $id"
    if ! python -c "import sys; from huggingface_hub import snapshot_download; \
snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])" "$id"; then
        if [ "$id" = facebook/sam3 ]; then
            echo "Skipped facebook/sam3: request access at https://huggingface.co/facebook/sam3 and run 'hf auth login'"
        else
            exit 1
        fi
    fi
done
du -sh "$HF_HOME"
