# material-vlm

Recognition pipeline that takes a prompt and a set of images and uses a
vision-language model (VLM) to report which materials are visible and what
condition they are in.

The model runs locally through Hugging Face `transformers`
(default: `Qwen/Qwen2.5-VL-7B-Instruct`), so it can run on an Euler GPU node
without any external API.

## Layout

```
configs/default.yaml      model, prompt file and generation settings
prompts/default.txt       default task prompt
src/material_vlm/
  cli.py                  command line entry point
  pipeline.py             image loading, prompt + images -> analysis
  frames.py               frame extraction from videos
  model.py                Hugging Face VLM wrapper
  schema.py               output schema, system prompt, JSON parsing
scripts/euler/
  env.sh                  modules, venv location, HF cache location
  setup_env.sh            one-time environment setup
  analyze.sbatch          Slurm job
tests/                    tests that run without a GPU or model download
```

## Setup on Euler

```bash
ssh <nethz>@euler.ethz.ch
git clone <repo-url> material-vlm && cd material-vlm
bash scripts/euler/setup_env.sh
```

This loads the modules in `scripts/euler/env.sh`, creates `.venv` in the
repository and installs the package. The Hugging Face cache is placed in
`$SCRATCH/hf_cache`; override `HF_HOME` or `VENV_DIR` before running the
script to change the locations.

## Running

### Images from a video

Extract one frame per second (the sharpest frame of each second is kept):

```bash
material-vlm-frames data/videos/site.mp4 --out data/frames/site --every 1.0
```

`--max-frames` limits the count and `--no-sharpest` keeps the first frame of
each interval instead. This runs on CPU, so a login node or your laptop is fine.

### Analysis

Put images in `data/` (ignored by git) and submit from the repository root:

```bash
sbatch scripts/euler/analyze.sbatch --images data/images --output outputs/run.json
```

Arguments after the script name are passed to `material-vlm`, on top of
`configs/default.yaml`:

```bash
# custom prompt
sbatch scripts/euler/analyze.sbatch --images data/images --output outputs/facade.json \
    --prompt "Focus on the facade cladding and describe any weathering."

# several views of the same object in one request
sbatch scripts/euler/analyze.sbatch --images data/object_01 --joint --output outputs/object_01.json
```

Logs go to `logs/`. The first run downloads the model weights (about 17 GB
for the default model).

For interactive debugging:

```bash
srun --gpus=1 --gres=gpumem:24g --cpus-per-task=4 --mem-per-cpu=8G --time=01:00:00 --pty bash
source scripts/euler/env.sh && source "$VENV_DIR/bin/activate"
material-vlm --config configs/default.yaml --images data/images/example.jpg
```

### Options

| Option | Meaning |
| --- | --- |
| `--images` | image files and/or directories (searched recursively) |
| `--prompt` / `--prompt-file` | task prompt, overrides `prompt_file` in the config |
| `--output` | JSON output file; prints to stdout if omitted |
| `--config` | YAML config file |
| `--model-id` | any model supported by `AutoModelForImageTextToText` |
| `--joint` / `--no-joint` | one request for all images / one request per image |
| `--max-image-side`, `--max-new-tokens`, `--temperature` | generation settings |

## Output

```json
{
  "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
  "prompt": "...",
  "results": [
    {
      "images": ["data/images/wall.jpg"],
      "analysis": {
        "summary": "A weathered brick wall with a steel door.",
        "materials": [
          {
            "name": "clay brick",
            "category": "masonry",
            "location": "wall",
            "confidence": 0.9,
            "condition": "fair",
            "defects": ["efflorescence"],
            "evidence": "red rectangular units with mortar joints"
          }
        ]
      },
      "parse_error": null,
      "raw_output": "..."
    }
  ]
}
```

`condition` is one of `good`, `fair`, `poor`, `unknown`. If the model output
cannot be parsed, `analysis` is `null`, `parse_error` explains why and
`raw_output` still holds the model's text.

The output structure is defined in `src/material_vlm/schema.py` (the
`SYSTEM_PROMPT` and the pydantic models); change both together to add fields.

## Tests

```bash
pytest
```

The tests use a fake model, so they need neither a GPU nor model weights.
