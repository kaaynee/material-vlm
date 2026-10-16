"""Box-prompted segmentation with SAM 2, plus box coordinate conversion."""

from __future__ import annotations

import numpy as np
from PIL import Image


def scale_boxes(
    boxes: list[list[float]],
    box_coords: str,
    input_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[list[float]]:
    """Convert VLM boxes into pixel coordinates of the target image.

    ``input_pixels`` boxes are relative to the image as the model saw it
    (``input_size``); ``normalized_1000`` boxes use a 0-1000 range per axis.
    Boxes are reordered, clipped to the image and dropped if empty.
    """
    width, height = target_size
    if box_coords == "normalized_1000":
        sx, sy = width / 1000, height / 1000
    elif box_coords == "input_pixels":
        sx, sy = width / input_size[0], height / input_size[1]
    else:
        raise ValueError(f"unknown box_coords: {box_coords}")

    scaled = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((min(max(x1 * sx, 0), width), min(max(x2 * sx, 0), width)))
        y1, y2 = sorted((min(max(y1 * sy, 0), height), min(max(y2 * sy, 0), height)))
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            scaled.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
    return scaled


class Segmenter:
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-large"):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(self.device).eval()

    def segment(
        self, image: Image.Image, boxes: list[list[float]]
    ) -> list[tuple[np.ndarray, float]]:
        """Return one (boolean mask, predicted IoU) per box, at image resolution."""
        if not boxes:
            return []
        inputs = self.processor(images=image, input_boxes=[boxes], return_tensors="pt")
        inputs = inputs.to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
        )[0]
        scores = outputs.iou_scores.cpu().flatten().tolist()
        return [(m[0].numpy().astype(bool), float(s)) for m, s in zip(masks, scores)]


# Text prompts SAM 3 is queried with for each defect type. "other" has none and
# falls back to box-prompted SAM 2.
DEFECT_CONCEPTS = {
    "crack": ["crack"],
    "corrosion": ["rust"],
    "biological_decay": ["moss", "algae", "lichen", "mold"],
}


class DefectSegmenter:
    """Text-prompted instance segmentation of defects with SAM 3."""

    def __init__(self, model_id: str = "facebook/sam3", threshold: float = 0.4):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self._torch = torch
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()

    def detect(self, image: Image.Image, defect_type: str) -> list[dict]:
        """Return all instances of a defect type as dicts with ``mask`` (bool
        array at image resolution), ``box`` (pixels), ``score`` and ``concept``."""
        detections = []
        for concept in DEFECT_CONCEPTS.get(defect_type, []):
            inputs = self.processor(images=image, text=concept, return_tensors="pt")
            inputs = inputs.to(self.device)
            with self._torch.inference_mode():
                outputs = self.model(**inputs)
            result = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.threshold,
                mask_threshold=0.5,
                target_sizes=inputs["original_sizes"].tolist(),
            )[0]
            for mask, box, score in zip(result["masks"], result["boxes"], result["scores"]):
                detections.append(
                    {
                        "mask": mask.cpu().numpy().astype(bool),
                        "box": [round(float(v), 1) for v in box],
                        "score": float(score),
                        "concept": concept,
                    }
                )
        return detections
