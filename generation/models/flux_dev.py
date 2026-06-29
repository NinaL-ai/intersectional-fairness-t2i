# generation/models/flux_dev.py

import torch
from diffusers import FluxPipeline
from .base import BaseGenerator


class FluxDevGenerator(BaseGenerator):

    def __init__(self):

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16
        )

        self.pipe.enable_model_cpu_offload()

    def generate(self, prompt, batch_size, seed):

        generator = torch.Generator(device="cpu").manual_seed(seed)

        return self.pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=24,
            max_sequence_length=256,
            generator=generator,
            num_images_per_prompt=batch_size
        ).images