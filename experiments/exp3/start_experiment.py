# experiments/exp1/start_experiment.py
import math
import os
import random

# from generation.models.loader import load_model
# from generation.prompts import PromptBuilder

# from classification.run import classify_model_images
from classification.postprocessing import postprocess_demographics

from evaluation import Metrics

from experiments.exp3.config import (PROFESSIONS, GROUPS, MODELS, NUM_IMAGES, BATCH_SIZE)


def save_images(images, model_name, profession, group, img_num, output_dir):
    path = os.path.join(output_dir, f"{model_name}/{profession}/neutral/")
    if not os.path.exists(path):
        os.makedirs(path)

    for i, image in enumerate(images):
        img_id = img_num + i

        filename = f"{profession}_{group}_{img_id}.png"
        image.save(os.path.join(path, filename))


def run_generation(output_dir):
    for model_name in MODELS:
        generator = load_model(model_name)
        for profession in PROFESSIONS:
            for run_id in range(math.ceil(NUM_IMAGES / BATCH_SIZE)):
                intersec_group = ", ".join(random.choice(values) for values in GROUPS.values())
                print(intersec_group)
                prompt = PromptBuilder.baseline(profession, intersec_group)
                img_num = run_id * BATCH_SIZE

                images = generator.generate(
                    prompt=prompt,
                    batch_size=BATCH_SIZE,
                    seed=img_num
                )

                save_images(
                    images,
                    model_name,
                    profession,
                    intersec_group,
                    img_num,
                    output_dir
                )


def main():
    exp_dir = "experiments/exp3/"
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
    #     exp="exp3"
    # )

    # 3. Postprocess
    postprocess_demographics(
        base_folder=raw_results_dir,
        output_folder=final_results_dir,
        exp="exp3"
    )

    # 4. fairness evaluation
    m = Metrics(base_dir="results/analysis_percentages")
    m.calc_intersectional_fairness()
    # m.plot_pairwise_js()



if __name__ == "__main__":
    main()

