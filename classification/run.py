# classification/run.py

import os
from classification.pipeline import ClassificationPipeline
from classification.analyzers.deepface_model import DeepFaceAnalyzer
from classification.analyzers.insightface_model import InsightFaceAnalyzer
from classification.analyzers.fairface_model import FairFaceAnalyzer


def classify_model_images(base_folder, output_folder, exp):
    pipeline = ClassificationPipeline(
        deepface=DeepFaceAnalyzer(),
        insightface=InsightFaceAnalyzer(),
        fairface=FairFaceAnalyzer()
    )

    for model in os.listdir(base_folder):
        model_path = os.path.join(base_folder, model)

        for specialty in os.listdir(model_path):

            specialty_path = os.path.join(model_path, specialty)
            if not os.path.isdir(specialty_path):
                continue

            for group in os.listdir(specialty_path):

                group_path = os.path.join(specialty_path, group)
                if not os.path.isdir(group_path):
                    continue

                print("Processing:", model, specialty, group)

                pipeline.run_group(
                    group_path,
                    os.path.join(output_folder, model, specialty),
                    model=model,
                    specialty=specialty,
                    group=group,
                    exp=exp
                )


