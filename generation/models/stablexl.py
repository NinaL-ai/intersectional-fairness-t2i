# generation/models/sdxl.py

import torch
from diffusers import DiffusionPipeline

from .base import BaseGenerator


class SDXLGenerator(BaseGenerator):

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )

        self.pipe = self.pipe.to(device)

    def generate(self, prompt, batch_size, seed):

        generator = torch.Generator(device="cpu").manual_seed(seed)

        return self.pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.0,
            generator=generator,
            num_images_per_prompt=batch_size
        ).images