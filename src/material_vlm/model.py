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
            model_id, torch_dtype="auto", device_map="auto"
        )
        self.model.eval()

    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
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
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
