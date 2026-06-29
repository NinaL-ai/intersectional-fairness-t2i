# experiments/exp1/start_experiment.py
import math
import os

# from generation.models.loader import load_model
# from generation.prompts import PromptBuilder

# from classification.run import classify_model_images
from classification.postprocessing import postprocess_demographics

from evaluation import Metrics

from experiments.exp2.config import (PROFESSIONS, GROUPS, MODELS, NUM_IMAGES, BATCH_SIZE)


def save_images(images, model_name, profession, group, img_num, output_dir, prompt):
    path = os.path.join(output_dir, f"images/{model_name}/{profession}/{group}/")
    if not os.path.exists(path):
        os.makedirs(path)

    for i, image in enumerate(images):
        img_id = img_num + i

        filename = f"{profession}_{group}_prompt_{prompt}_{img_id}.png"
        image.save(os.path.join(path, filename))


def run_generation(output_dir):
    for model_name in MODELS:
        generator = load_model(model_name)
        for profession in PROFESSIONS:
            for group in GROUPS:
                if group in ["20 years old", "40 years old", "75 years old"]:
                    prompts = PromptBuilder.baseline(profession, group)
                else:
                    prompts = PromptBuilder.sensitivity(profession, group)
                for run_id in range(math.ceil(NUM_IMAGES / BATCH_SIZE)):
                    img_num = run_id * BATCH_SIZE
                    for i, prompt in enumerate(prompts):
                        images = generator.generate(
                            prompt=prompt,
                            batch_size=BATCH_SIZE,
                            seed=img_num
                        )

                        save_images(
                            images,
                            model_name,
                            profession,
                            group,
                            img_num,
                            output_dir,
                            i
                        )


def main():
    # exp_dir = "experiments/exp2/"
    exp_dir = ""

    image_dir = os.path.join(exp_dir, "images/")
    raw_results_dir = os.path.join(exp_dir, "results/analysis_results_raw/")
    final_results_dir = os.path.join(exp_dir, "results/analysis_results_merged/")

    # 1. Generate images
    # run_generation(image_dir)

    # 2. Classification
    # classify_model_images(
    #     base_folder=image_dir,
    #     output_folder=raw_results_dir,
    #     exp="exp2"
    # )

    # 3. Postprocess
    # postprocess_demographics(
    #     base_folder=raw_results_dir,
    #     output_folder=final_results_dir,
    #     exp="exp2"
    # )

    # 4. fairness evaluation
    m = Metrics(base_dir="results/analysis_percentages_age", out_path="results/fairness_age/",
                jsd_path="results/dependency_matrix_age/")
    m.calc_intersectional_fairness()



if __name__ == "__main__":
    main()

