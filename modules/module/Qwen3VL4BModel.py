from modules.module.BaseImageCaptionModel import BaseImageCaptionModel, CaptionSample

import torch

from transformers import AutoModelForImageTextToText, AutoProcessor


class Qwen3VL4BModel(BaseImageCaptionModel):
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = self.__supported_device(device)
        self.dtype = self.__supported_dtype(self.device, dtype)
        self.model_id = "Qwen/Qwen3-VL-4B-Instruct"

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        )
        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def __supported_device(device: torch.device) -> torch.device:
        if device.type == "cuda":
            if not torch.cuda.is_available():
                print("CUDA is not available. Loading Qwen3-VL 4B on CPU instead.")
                return torch.device("cpu")

            if device.index is not None and device.index >= torch.cuda.device_count():
                print(f"CUDA device {device} is not available. Loading Qwen3-VL 4B on CPU instead.")
                return torch.device("cpu")

        if device.type == "mps" and not torch.backends.mps.is_available():
            print("MPS is not available. Loading Qwen3-VL 4B on CPU instead.")
            return torch.device("cpu")

        return device

    @staticmethod
    def __supported_dtype(device: torch.device, dtype: torch.dtype) -> torch.dtype:
        if device.type == "cpu":
            return torch.float32

        if device.type == "cuda":
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                return torch.float16
            if dtype in (torch.float16, torch.bfloat16, torch.float32):
                return dtype
            return torch.float16

        if device.type == "mps":
            if dtype in (torch.float16, torch.float32):
                return dtype
            return torch.float16

        if dtype in (torch.float16, torch.bfloat16, torch.float32):
            return dtype
        return torch.float32

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
        predicted_caption = (caption_prefix + predicted_caption + caption_postfix).strip()

        return predicted_caption
