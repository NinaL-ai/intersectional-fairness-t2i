# classification/analyzers/fairface_analyzer.py

import os
from classification.analyzers.fairface import predict_bbox


class FairFaceAnalyzer:
    def predict_folder(self, image_paths, output_path, model, specialty, group, exp):

        save_dir = f"classification/analyzers/fairface/detected_faces/{exp}/{model}_{specialty}_{group}/"
        os.makedirs(save_dir, exist_ok=True)

        bboxes = predict_bbox.detect_face(image_paths, save_dir)

        predict_bbox.predidct_age_gender_race(
            output_path,
            bboxes,
            save_dir
        )