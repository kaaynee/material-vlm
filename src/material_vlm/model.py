"""Thin wrapper around a Hugging Face image-text-to-text model."""

from __future__ import annotations

from PIL import Image


class VLM:
    def __init__(self, model_id: str):
        # Imported here so the rest of the package works without a GPU stack.
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype="auto", device_map="auto"
        )
        self.model.eval()

    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> tuple[str, list[tuple[int, int]]]:
        """Return the generated text and the (width, height) of each image as the
        model saw it, which is the coordinate frame of any boxes it outputs."""
        messages = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        sampling = (
            {"do_sample": True, "temperature": temperature}
            if temperature > 0
            else {"do_sample": False}
        )
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, **sampling
            )
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        return text, self._input_sizes(inputs, images)

    def _input_sizes(self, inputs, images) -> list[tuple[int, int]]:
        # Qwen-style processors resize images to a grid of patches; boxes refer
        # to that resized image, not to the image we passed in.
        grid = inputs.get("image_grid_thw")
        patch = getattr(self.processor.image_processor, "patch_size", None)
        if grid is None or patch is None:
            return [image.size for image in images]
        return [(int(w) * patch, int(h) * patch) for _, h, w in grid.tolist()]
