# classification/pipeline.py

import os
import pandas as pd
from tqdm import tqdm


class ClassificationPipeline:

    def __init__(self, deepface, insightface, fairface):
        self.deepface = deepface
        self.insightface = insightface
        self.fairface = fairface

    def run_group(self, group_path, output_path, model, specialty, group, exp):
        print("group_path: ", group_path)

        deep_rows = []
        ins_rows = []
        image_paths = []

        for img_file in tqdm(os.listdir(group_path), desc=f"{specialty}-{group}"):

            img_path = os.path.join(group_path, img_file)
            image_paths.append(img_path)

            try:
                deep_rows.append(self.deepface.predict(img_path, img_file))
                ins_rows.append(self.insightface.predict(img_path, img_file))
            except Exception as e:
                print(f"Skip {img_file}: {e}")

        # save DeepFace
        df_deep = pd.DataFrame(deep_rows)
        df_ins = pd.DataFrame(ins_rows)

        os.makedirs(output_path, exist_ok=True)

        df_deep.to_csv(os.path.join(output_path, f"{group}_deep.csv"), index=False)
        df_ins.to_csv(os.path.join(output_path, f"{group}_ins.csv"), index=False)

        # FairFace (folder-level)
        ff_out = os.path.join(output_path, f"{group}_fairface.csv")
        self.fairface.predict_folder(
            image_paths,
            ff_out,
            model,
            specialty,
            group,
            exp
        )