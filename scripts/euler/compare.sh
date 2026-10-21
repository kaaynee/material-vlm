#!/bin/bash
# Submit one job per model on the same images. Run from the repository root:
#   bash scripts/euler/compare.sh <images> <name> [set] [extra material-vlm args...]
#
#   bash scripts/euler/compare.sh data/frames/site site            # small set
#   bash scripts/euler/compare.sh data/frames/site site large      # largest models per family
#   bash scripts/euler/compare.sh data/frames/site site "qwen3-vl-32b qwen3.5-27b"
#   TIME_LIMIT=12:00:00 bash scripts/euler/compare.sh ...       # override time limits
#
# Results land in outputs/<name>/<model>.json; afterwards compare them with
#   material-vlm-compare outputs/<name>/*.json --out outputs/<name>/comparison
# Download the weights first (scripts/euler/download.sbatch) so GPU time is not
# spent downloading.
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,13p' "$0" >&2
    exit 1
fi
images=$1
name=$2
shift 2
if [ ! -e "$images" ]; then
    echo "image path does not exist: $images (relative paths are resolved from $(pwd))" >&2
    exit 1
fi
set_name=small
if [ $# -gt 0 ] && [[ $1 != -* ]]; then
    set_name=$1
    shift
fi

case $set_name in
    small) models="qwen2.5-vl-7b qwen3-vl-8b qwen3.5-9b" ;;
    large) models="qwen2.5-vl-72b qwen3-vl-32b qwen3.5-27b" ;;
    moe)   models="qwen3-vl-30b-a3b qwen3.5-35b-a3b qwen3.5-122b-a10b" ;;
    *)     models=$set_name ;;
esac

# Resources per model: <number of GPUs> <memory per GPU> <time limit>.
# bf16 weights take about 2 GB per billion parameters, plus room for images and SAM 2.
resources() {
    case $1 in
        qwen2.5-vl-7b)     echo "1 24g 01:00:00" ;;
        qwen3-vl-8b)       echo "1 24g 01:00:00" ;;
        qwen3.5-9b)        echo "1 32g 01:00:00" ;;
        qwen3.5-27b)       echo "1 80g 03:00:00" ;;  # 28B,  ~56 GB
        qwen3-vl-30b-a3b)  echo "1 80g 02:00:00" ;;  # 31B,  ~62 GB, MoE: fast
        qwen3-vl-32b)      echo "1 80g 03:00:00" ;;  # 33B,  ~67 GB
        qwen3.5-35b-a3b)   echo "2 80g 02:00:00" ;;  # 36B,  ~72 GB, MoE: fast
        qwen2.5-vl-72b)    echo "2 80g 04:00:00" ;;  # 73B, ~147 GB
        qwen3.5-122b-a10b) echo "4 80g 04:00:00" ;;  # 125B, ~250 GB
        *) echo "no resources defined for $1 in $0" >&2; return 1 ;;
    esac
}

for model in $models; do
    if [ ! -f "configs/models/$model.yaml" ]; then
        echo "missing configs/models/$model.yaml" >&2
        exit 1
    fi
    read -r gpus gpumem time_limit <<< "$(resources "$model")"
    time_limit=${TIME_LIMIT:-$time_limit}
    echo "Submitting $model ($gpus GPU x $gpumem, $time_limit)"
    sbatch --job-name="mvlm-$model" --gpus="$gpus" --gres="gpumem:$gpumem" \
        --time="$time_limit" --mem-per-cpu=16G \
        scripts/euler/analyze.sbatch \
        --config "configs/models/$model.yaml" \
        --images "$images" \
        --output "outputs/$name/$model.json" \
        "$@"
done
