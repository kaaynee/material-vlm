# material-vlm

Recognition pipeline that takes a prompt and a set of images and uses a
vision-language model (VLM) to report which materials are visible and what
condition they are in.

The model runs locally through Hugging Face `transformers`
(default: `Qwen/Qwen2.5-VL-7B-Instruct`), so it can run on an Euler GPU node
without any external API. Per image it produces a JSON analysis, pixel masks
for every material and defect (via SAM 2) and overlay images showing where
everything was detected.

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
  segmentation.py         SAM 2 masks from boxes, box coordinate conversion
  visualize.py            overlay images
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
| `--segment` / `--no-segment` | masks with SAM 2 / boxes only |
| `--segmenter-id` | SAM 2 checkpoint (default `facebook/sam2.1-hiera-large`) |
| `--box-coords` | `input_pixels` (Qwen2.5-VL) or `normalized_1000` (Qwen3-VL) |

## Output

For `--output outputs/run.json` the pipeline writes:

```
outputs/run.json                         analysis of all images
outputs/run/overlays/0000_wall_materials.jpg   materials, colored per material, labeled with condition
outputs/run/overlays/0000_wall_defects.jpg     defects, colored by type
outputs/run/masks/0000_wall/material_00_clay-brick.png
outputs/run/masks/0000_wall/defect_00_00_crack.png
```

Masks are binary PNGs (0/255) at the original image resolution. The JSON
looks like this (one entry in `results` per image):

```json
{
  "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
  "prompt": "...",
  "results": [
    {
      "images": ["data/images/wall.jpg"],
      "analysis": {
        "summary": "A brick wall next to a steel plate.",
        "materials": [
          {
            "name": "clay brick",
            "category": "masonry",
            "boxes": [[40.0, 60.0, 700.0, 740.0]],
            "confidence": 0.9,
            "condition": "fair",
            "evidence": "red rectangular units with mortar joints",
            "defects": [
              {
                "type": "crack",
                "severity": "medium",
                "description": "diagonal crack through several bricks",
                "boxes": [[190.0, 110.0, 270.0, 460.0]],
                "mask": "outputs/run/masks/0000_wall/defect_00_00_crack.png",
                "mask_score": 0.84,
                "area_fraction": 0.0035
              }
            ],
            "mask": "outputs/run/masks/0000_wall/material_00_clay-brick.png",
            "mask_score": 0.97,
            "area_fraction": 0.3723
          }
        ]
      },
      "overlays": {
        "materials": "outputs/run/overlays/0000_wall_materials.jpg",
        "defects": "outputs/run/overlays/0000_wall_defects.jpg"
      },
      "parse_error": null,
      "raw_output": "..."
    }
  ]
}
```

- `condition`: `good`, `fair`, `poor` or `unknown`.
- Defect `type`: `crack`, `corrosion`, `biological_decay` (mold, moss, algae,
  lichen, fungi, rot) or `other`; `severity`: `low`, `medium`, `high` or `unknown`.
- `boxes`: `[x1, y1, x2, y2]` in pixels of the original image.
- `mask_score`: SAM 2's predicted mask quality (0-1); `area_fraction`: share
  of the image covered by the mask.

If the model output cannot be parsed, `analysis` is `null`, `parse_error`
explains why and `raw_output` still holds the model's text.

### How localization works

The VLM outputs a bounding box per material region and per defect area
(Qwen2.5-VL is trained for this kind of grounding). The boxes are converted
from the model's internal image size to the original image and passed to
SAM 2 as prompts, which returns a pixel mask per box. Masks of the same
material or defect are merged.

- Box accuracy is the limiting factor: SAM 2 segments whatever is most
  prominent in the box, so a loose box around a crack can yield a mask of the
  surrounding surface. Check `mask_score` and the overlays.
- Boxes and masks are only produced per image. With `--joint` they are left
  empty, since they cannot be attributed to a single image.
- `--no-segment` skips SAM 2 and draws boxes only. For a VLM that uses 0-1000
  normalized coordinates (e.g. Qwen3-VL), set `--box-coords normalized_1000`.

The output structure is defined in `src/material_vlm/schema.py` (the system
prompt and the pydantic models); change both together to add fields.

## Tests

```bash
pytest
```

The tests use a fake model, so they need neither a GPU nor model weights.
