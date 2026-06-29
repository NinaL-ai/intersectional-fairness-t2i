from deepface_model import DeepFace
import insightface_model
from insightface_model.app import FaceAnalysis
import cv2
import os
from tqdm import tqdm
import pandas as pd
from classification.analyzers.fairface_model import predict_bbox


def analyse_deepface(img_path, img_file):
    # gender, race, emotion detection
    result = DeepFace.analyze(
        img_path,
        actions=['age', 'gender', 'race', 'emotion'],
        enforce_detection=False,
        detector_backend="retinaface")
    result = result[0]

    # result is a dict
    row = {}
    row['face_confidence_deepface'] = result.get('face_confidence', None)
    row["age_deepface"] = result.get("age", None)
    row["gender_deepface"] = result.get('dominant_gender', None)
    gender_dict = result.get("gender", {})
    for gender_name, gender_value in gender_dict.items():
        row[gender_name] = gender_value

    row["race_deepface"] = result.get('dominant_race', None)
    race_dict = result.get("race", {})
    for race_name, race_value in race_dict.items():
        row[race_name] = race_value

    row["emotion_deepface"] = result.get("dominant_emotion", None)
    emotion_dict = result.get("emotion", {})
    for emotion_name, emotion_value in emotion_dict.items():
        row[emotion_name] = emotion_value

    row["image"] = img_file

    return row


def analyse_insightface(app, img_path, img_file):
    img = cv2.imread(img_path)
    faces = app.get(img)

    row = {}
    if len(faces) == 0:
        img = cv2.resize(img, None, fx=2, fy=2)
        faces = app.get(img)
    if len(faces) == 0:
        print("no faces detected for ", img_path)
    elif len(faces) > 0:
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        row["age_insightface"] = face.age
        row["gender_insightface"] = face.gender

    row["image"] = img_file

    return row


def analyse_fairface(group_imgs, output_path, model, specialty, group):
    dir = model + "_" + specialty + "_" + group
    SAVE_DETECTED_AT = f"fairface/detected_faces/{dir}/"
    os.makedirs(SAVE_DETECTED_AT, exist_ok=True)
    bboxes = predict_bbox.detect_face(group_imgs, SAVE_DETECTED_AT)
    predict_bbox.predidct_age_gender_race(output_path, bboxes, SAVE_DETECTED_AT)


def analyse_model_images(base_folder, output_folder):
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0) #, det_size=(1024,1024))

    # Loop over specialties (subfolders)
    for specialty in os.listdir(base_folder):
        specialty_path = os.path.join(base_folder, specialty)
        if not os.path.isdir(specialty_path):
            continue

        for group in os.listdir(specialty_path):
            group_imgs = []
            group_path = os.path.join(specialty_path, group)
            if not os.path.isdir(group_path):
                continue
            print("analyse model ", base_folder, ", specialty ", specialty_path, ", group ", group_path)

            data_deepface, data_insightface = [], []
            for img_file in tqdm(os.listdir(group_path), desc=f"Processing {specialty}"):
                img_path = os.path.join(group_path, img_file)
                group_imgs.append(img_path)
                try:
                    res_deepface = analyse_deepface(img_path, img_file)
                    res_insightface = analyse_insightface(app, img_path, img_file)
                    data_deepface.append(res_deepface)
                    data_insightface.append(res_insightface)

                except Exception as e:
                    print(f"Skipping {img_file} due to error: {e}")

            output_specialty_path = os.path.join(output_folder, specialty)
            os.makedirs(output_specialty_path, exist_ok=True)

            # Save DataFrame for this group
            df_deepface = pd.DataFrame(data_deepface)
            df_insightface = pd.DataFrame(data_insightface)
            output_path_deep = os.path.join(output_specialty_path, f"{group}_deep.csv")
            df_deepface.to_csv(output_path_deep, index=False)
            output_path_ins = os.path.join(output_specialty_path, f"{group}_ins.csv")
            df_insightface.to_csv(output_path_ins, index=False)

            # fairface prediction
            output_path_fairface = os.path.join(output_specialty_path, f"{group}_fairface.csv")
            analyse_fairface(group_imgs, output_path_fairface, model, specialty, group)

            print(f"Saved analysis for {model} {specialty} {group}")


if __name__ == "__main__":
    # Folder containing images per specialty
    exp_dir = "../experiments/exp1/"
    models = [
        "Stablexl1.0",
        "Stable3.5-medium",
        "Flux-dev"
    ]

    for model in models:
        base_folder = exp_dir + f"images/{model}/"
        output_folder = exp_dir + f"results/analysis_results_raw/{model}/"  # Folder to save CSVs
        os.makedirs(output_folder, exist_ok=True)
        analyse_model_images(base_folder, output_folder)
