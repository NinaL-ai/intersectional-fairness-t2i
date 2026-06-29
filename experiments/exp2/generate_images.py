import math

from diffusers import DiffusionPipeline
from diffusers import PixArtSigmaPipeline
from diffusers import Flux2KleinPipeline
import torch
import os
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from diffusers import FluxPipeline

def save_result(images, model, specialty, group, offset, num_image, dir, prompt):
    path = os.path.join(dir, f"images/{model}/{specialty}/{group}/")
    if not os.path.exists(path):
        os.makedirs(path)

    for i, image in enumerate(images):
        img_id = num_image + offset + i

        filename = f"{specialty}_{group}_prompt_{prompt}_{img_id}.png"
        image.save(os.path.join(path, filename))


def run_single_experiment(specialty, group, model, pipe, offset, batch_size, run, dir):
    try:
        num_image = run * batch_size
        print(f"---------------------- {model} {specialty} {group} {num_image} ----------------------------")
        if group == "neutral":
            # baseline (f"Photo portrait of a {specialty}, centered, head-and-shoulders framing, looking at the camera, photorealistic"),
            prompts = [(f"Professional photograph of a {specialty}, centered, head-and-shoulders framing, looking at the camera, photorealistic"),
                       (f"Portrait photograph of a {specialty}, centered, head-and-shoulders framing, looking at the camera, photorealistic")
                       ]
        else:
            # prompt = (f"Photo portrait of a {group} {specialty}, centered, head-and-shoulders framing, looking at the camera, photorealistic")
            prompts = [(f"Photo portrait of a {specialty} who is {group}, centered, head-and-shoulders framing, looking at the camera, photorealistic"),
                       (f"Photo portrait of a {specialty}. The person is {group}. Centered, head-and-shoulders framing, looking at the camera, photorealistic")
                       ]

        # negative_prompt = "lowres, child, illustration, covered face, hidden face"

        for i, prompt in enumerate(prompts):
            seed = num_image + offset
            generator = torch.Generator(device="cpu").manual_seed(seed)

            if model in ["Flux2-klein", "Flux-schnell"]:
                # generator = torch.Generator(device="cpu" if model == "Flux" else pipe.device).manual_seed(num_image)
                image = pipe(prompt=prompt,
                             guidance_scale=1.0,
                             num_inference_steps=4,
                             generator=generator
                             ).images[0]
            elif model == "Flux-dev":
                images = pipe(
                    prompt,
                    height=1024,  # 768 if too slow
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=24,
                    max_sequence_length=256,
                    generator=generator,
                    num_images_per_prompt=batch_size
                ).images  #[0]
            elif model == "Stable3.5-medium":
                images = pipe(
                    prompt=prompt,
                    num_inference_steps = 25,
                    guidance_scale = 4.5,
                    max_sequence_length = 256,
                    generator=generator,
                    num_images_per_prompt=batch_size
                ).images
            elif model == "Stablexl1.0":
                images = pipe(
                    prompt=prompt,
                    num_inference_steps = 30,
                    guidance_scale = 7.0,
                    generator=generator,
                    num_images_per_prompt=batch_size
                ).images

            save_result(images, model, specialty, group, offset, num_image, dir, i)
    except Exception as e:
        print(f"\n Experiment failed for specialty='{specialty}', num_image={num_image}")
        raise


def start_experiment(specialties, groups, models, dir, num_images, offset=0, batch_size=4):
    token = "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"

    for model in models:
        if model == "Stablexl1.0":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                     torch_dtype=torch.float16,
                                                     use_safetensors=True,
                                                     variant="fp16")
            pipe = pipe.to(device)
        elif model == "Stable3.5-medium":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-medium",
                torch_dtype=torch.float16,  # torch.bfloat16
                use_safetensors=True
            )
            pipe = pipe.to(device)
        elif model == "Flux-dev":
            # device = "cuda" #torch.bfloat16)
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",torch_dtype=torch.float16)
            pipe.enable_model_cpu_offload()
        elif model == "Pixart":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pipe = PixArtSigmaPipeline.from_pretrained(
                "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            pipe = pipe.to(device)

        # not used
        elif model == "Stable3.5-large":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                use_safetensors=True
            )
            pipe = pipe.to(device)
        elif model == "Stablebase":
            device = "cuda"
            model_id = "Manojb/stable-diffusion-2-1-base"
            scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16,
                                                           token=token)
        elif model == "Flux2-klein":
            pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16)
            pipe.enable_model_cpu_offload()
        elif model == "Flux-schnell":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = DiffusionPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                use_safetensors=True
            )
            pipe.enable_model_cpu_offload()

        for specialty in specialties:
            for group in groups:
                runs = math.ceil(num_images / batch_size)
                for run in range(runs):
                    run_single_experiment(specialty, group, model, pipe, offset, batch_size, run, dir)





if __name__ == "__main__":
    # Experiment 2: Sensitivity analysis to prompts
    professions = [
        # "doctor", # Healthcare
        "writer",  # Creative
        "judge",  # Law / authority
        "software engineer",  # Tech / science
        "teacher",  # Education
        "CEO",  # Business / white collar
        "chef",  # Service / hospitality  #
        "cashier",  # Manual labor
        "professional athlete"  # Sports
    ]

    groups = [
        "neutral",
        "female", "male",
        "young", "middle aged", "elderly",
        "white", "asian", "indian", "black"
    ]
    # groups = ["20 years old", "40 years old", "75 years old"]

    # Available models: "Stablexl1.0", "Pixart", "Flux", "Stablebase", "OpenJ", "Stable3.5-medium", "Stable3.5-large", "Flux-schnell"
    models = [
        "Stablexl1.0",
        "Stable3.5-medium",
        "Flux-dev"
    ]

    dir = ""
    start_experiment(professions, groups, models, dir, num_images=50, offset=0, batch_size=5)
