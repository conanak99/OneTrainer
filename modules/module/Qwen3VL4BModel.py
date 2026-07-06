from modules.module.BaseImageCaptionModel import BaseImageCaptionModel, CaptionSample

import torch

from transformers import AutoModelForImageTextToText, AutoProcessor


class Qwen3VL4BModel(BaseImageCaptionModel):
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.model_id = "Qwen/Qwen3-VL-4B-Instruct"

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        )
        self.model.eval()
        self.model.to(self.device)

    def generate_caption(
            self,
            caption_sample: CaptionSample,
            initial_caption: str = "",
            caption_prefix: str = "",
            caption_postfix: str = "",
    ) -> str:
        prompt = initial_caption

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": caption_sample.get_image()},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids
            in zip(inputs.input_ids, generated_ids)
        ]
        predicted_caption = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        predicted_caption = (
            caption_prefix + initial_caption + predicted_caption + caption_postfix
        ).strip()

        return predicted_caption
