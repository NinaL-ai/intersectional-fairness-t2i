# generation/models/stable35.py

import torch
from diffusers import DiffusionPipeline

from .base import BaseGenerator


class Stable35Generator(BaseGenerator):

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        self.pipe = self.pipe.to(device)

    def generate(self, prompt, batch_size, seed):

        generator = torch.Generator(device="cpu").manual_seed(seed)

        return self.pipe(
            prompt=prompt,
            num_inference_steps=25,
            guidance_scale=4.5,
            max_sequence_length=256,
            generator=generator,
            num_images_per_prompt=batch_size
        ).images