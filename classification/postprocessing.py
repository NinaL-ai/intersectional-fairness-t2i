import itertools
import os

import numpy as np
import pandas as pd
import re


def calc_group_percentage(exp):
    input_root = "results/analysis_results_merged"
    output_root = "results/analysis_percentages"

    os.makedirs(output_root, exist_ok=True)

    for model in os.listdir(input_root):
        model_path = os.path.join(input_root, model)

        if not os.path.isdir(model_path):
            continue

        for specialty in os.listdir(model_path):
            specialty_path = os.path.join(model_path, specialty)

            for file in os.listdir(specialty_path):

                if not file.endswith(".csv"):
                    continue

                group = file.replace(".csv", "")
                path = os.path.join(specialty_path, file)

                df = pd.read_csv(path)
                if exp == "exp2" and group not in ["20 years old", "40 years old", "75 years old"]:
                    df_prompt_0 = df[df["image"].str.contains(r"prompt_0_", na=False)].copy()
                    df_prompt_1 = df[df["image"].str.contains(r"prompt_1_", na=False)].copy()
                    dfs = {"prompt0": df_prompt_0, "prompt1": df_prompt_1}
                else:
                    dfs = {"prompt base": df}

                for key, df in dfs.items():
                    results = []
                    total = len(df)

                    # age
                    age_counts = df["age"].value_counts(dropna=False)
                    for age, count in age_counts.items():
                        results.append({
                            "group": "age",
                            "category": age,
                            "percentage": count / total * 100
                        })

                    # gender
                    gender_counts = df["gender"].value_counts(dropna=False)
                    for gender, count in gender_counts.items():
                        results.append({
                            "group": "gender",
                            "category": gender,
                            "percentage": count / total * 100
                        })

                    # race
                    race_counts = df["race"].value_counts(dropna=False)
                    for race, count in race_counts.items():
                        results.append({
                            "group": "race",
                            "category": race,
                            "percentage": count / total * 100
                        })

                    # Intersectional groups
                    inter_counts = df.groupby(["age", "gender", "race"]).size()

                    for (age, gender, race), count in inter_counts.items():
                        results.append({
                            "group": "intersection",
                            "category": f"{age}_{gender}_{race}",
                            "percentage": count / total * 100
                        })

                    result_df = pd.DataFrame(results)

                    if key != "prompt base":
                        specialty_out = os.path.join(output_root + "_" + key, model, specialty)
                    else:
                        specialty_out = os.path.join(output_root, model, specialty)

                    out_file = os.path.join(specialty_out, f"{group}_percentages.csv")
                    os.makedirs(specialty_out, exist_ok=True)
                    result_df.to_csv(out_file, index=False)

                    print(f"Saved {out_file}")



def fill_missing_groups():
    base_dir = "results/analysis_percentages/"

    age_groups = ["Young", "Middle", "Old"]
    gender_groups = ["Female", "Male"]
    race_groups = ["White", "Asian", "Indian", "Black"]

    required_groups = {
        "gender": gender_groups,
        "age": age_groups,
        "race": race_groups
    }

    # build all intersection labels
    all_intersections = [
        f"{a}_{g}_{r}"
        for a, g, r in itertools.product(age_groups, gender_groups, race_groups)
    ]

    for model in os.listdir(base_dir):

        model_path = os.path.join(base_dir, model)

        if not os.path.isdir(model_path):
            continue

        for specialty in os.listdir(model_path):
            specialty_path = os.path.join(model_path, specialty)

            for file in os.listdir(specialty_path):

                if not file.endswith(".csv"):
                    continue

                path = os.path.join(specialty_path, file)
                df = pd.read_csv(path)

                rows_to_add = []

                # ---- fill marginal groups ----
                for group, categories in required_groups.items():

                    existing = df[df["group"] == group]["category"].tolist()

                    for cat in categories:
                        if cat not in existing:
                            rows_to_add.append({
                                "group": group,
                                "category": cat,
                                "percentage": 0.0
                            })

                # fill intersection groups
                existing_intersections = df[df["group"] == "intersection"]["category"].tolist()

                for inter in all_intersections:
                    if inter not in existing_intersections:
                        rows_to_add.append({
                            "group": "intersection",
                            "category": inter,
                            "percentage": 0.0
                        })

                if rows_to_add:
                    df = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)

                df = df.sort_values(["group", "category"]).reset_index(drop=True)

                df.to_csv(path, index=False)

        print("All missing groups and intersections added.")



# Extract original image filename from face_name_align
def extract_image_name(path):
    filename = os.path.basename(path)

    # remove "_faceX" before extension
    return re.sub(r'_face\d+(\.[^.]+)$', r'\1', filename)


def parse_bbox(bbox_str):
    # find all (x, y) pairs
    coords = re.findall(r'\((-?\d+),\s*(-?\d+)\)', bbox_str)

    # convert to numpy array
    return np.array([int(coords[0][0]), int(coords[0][1]),
                     int(coords[1][0]), int(coords[1][1])], dtype=np.int32)


def iou(boxA, boxB):
    # intersection coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter_area

    return inter_area / union if union > 0 else 0.0


def deduplicate_faces(df, iou_thresh=0.8):
    df = df.copy()
    keep = []

    for img, group in df.groupby("image"):
        group = group.reset_index(drop=True)
        used = set()

        for i in range(len(group)):
            if i in used:
                continue

            box_i = group.loc[i, "bbox"]
            cluster = [i]

            for j in range(i+1, len(group)):
                if j in used:
                    continue

                box_j = group.loc[j, "bbox"]
                if iou(box_i, box_j) > iou_thresh:
                    cluster.append(j)
                    used.add(j)

            # keep the largest box in cluster (more stable)
            boxes = group["bbox"].values
            best = max(
                cluster,
                key=lambda k: (boxes[k][2] - boxes[k][0]) * (boxes[k][3] - boxes[k][1])
            )
            keep.append(group.loc[best])

    return pd.DataFrame(keep)


def extract_demographics(input_folder, output_folder):
    """
    # deepface: Gender, Age exact, race (white, black, asian, indian, middle eastern, latino hispanic)
    # fairface: Gender, Age in bins, race7 (White, Black, Latino_Hispanic, East, Southeast Asian, Indian, Middle Eastern)
    # or race4 (White, Black, Asian and Indian)
    # Insightface: Gender, Age exact
    """
    # Folders
    os.makedirs(output_folder, exist_ok=True)

    # Age bins
    bins = [0, 29, 60, 200]  # Young: 0–29, Middle: 30–60, Old: 61+
    labels = ["Young", "Middle", "Old"]
    count = []
    count_age = []
    count_race = []

    # Loop over each model
    for model_name in os.listdir(input_folder):
        model_path = os.path.join(input_folder, model_name)
        for specialty in os.listdir(model_path):
            specialty_path = os.path.join(model_path, specialty)

            if os.path.isdir(specialty_path):
                # Create corresponding output subfolder
                output_specialty_path = os.path.join(output_folder, model_name, specialty)
                os.makedirs(output_specialty_path, exist_ok=True)

                # Loop over CSVs for each specialty
                for file in os.listdir(specialty_path):
                    if file.endswith("fairface.csv"):
                        print(f"Processing {specialty_path} {file}")
                        group = file.split("_")[0]
                        # results = {}
                        df_ff = pd.read_csv(os.path.join(specialty_path, file))
                        df_df = pd.read_csv(os.path.join(specialty_path, group + "_deep.csv"))
                        df_if = pd.read_csv(os.path.join(specialty_path, group + "_ins.csv"))

                        # rename columns
                        # df_ff = df_ff.add_suffix("_fair")
                        df_ff.rename(columns={"face": "face_fair", "age": "age_fair", "gender": "gender_fair",
                                              "race4": "race_fair"}, inplace=True)
                        df_df["gender_scores_deep"] = (df_df[["gender_Man", "gender_Woman"]].div(100).values.tolist())
                        df_ff["face_fair"] = (df_ff["face_name_align"].str.extract(r'_face(\d+)\.').astype(int))
                        df_if["gender_insightface"] = df_if["gender_insightface"].map({1.0: "Male", 0.0: "Female"})
                        df_df["gender_deepface"] = df_df["gender_deepface"].map({"Man": "Male", "Woman": "Female"})

                        # # Create age bins
                        # df['age_bin'] = pd.cut(df['age_insightface'], bins=bins, labels=labels, right=True)
                        age_map = {
                            "0-2": 1,
                            "3-9": 6,
                            "10-19": 15,
                            "20-29": 25,
                            "30-39": 35,
                            "40-49": 45,
                            "50-59": 55,
                            "60-69": 65,
                            "70+": 75,
                        }
                        df_ff["age_num_fair"] = df_ff["age_fair"].map(age_map)
                        df_ff["age_bin_fair"] = pd.cut(
                            df_ff["age_fair"].map(age_map),
                            bins=bins,
                            labels=labels,
                            include_lowest=True
                        )
                        df_if["age_bin_insightface"] = pd.cut(
                            df_if["age_insightface"],
                            bins=bins,
                            labels=labels,
                            include_lowest=True
                        )
                        df_df["age_bin_deepface"] = pd.cut(
                            df_df["age_deepface"],
                            bins=bins,
                            labels=labels,
                            include_lowest=True
                        )

                        race_map = {
                            "white": "White",
                            "black": "Black",
                            "asian": "Asian",
                            "indian": "Indian",
                            "middle eastern": "White",
                            "latino hispanic": "White"
                        }

                        df_df["race4_deepface"] = df_df["race_deepface"].map(race_map)

                        # Merge dataframes
                        df_ff["image"] = df_ff["face_name_align"].apply(extract_image_name)
                        df_base = pd.merge(df_ff, df_df, on="image", how="inner")
                        df_merged = pd.merge(df_base, df_if, on="image", how="inner")

                        df_merged["equal_gender"] = (
                            df_merged[["gender_fair", "gender_deepface", "gender_insightface"]]
                            .apply(lambda row: row.dropna().nunique() == 1, axis=1)
                        )
                        df_merged["equal_age"] = (
                            df_merged[["age_bin_fair", "age_bin_deepface"]]  # insightface
                            .apply(lambda row: row.dropna().nunique() == 1, axis=1)
                        )
                        df_merged["equal_race"] = (
                            df_merged[["race_fair", "race4_deepface"]]  # insightface
                            .apply(lambda row: row.dropna().nunique() == 1, axis=1)
                        )

                        df_merged["age_diff"] = abs(df_merged["age_num_fair"] - df_merged["age_deepface"])

                        df_merged["race"] = df_merged["race_fair"]
                        df_merged["age"] = df_merged["age_bin_fair"]
                        df_merged["gender"] = df_merged.apply(
                            lambda row: row["gender_fair"] if row["equal_gender"] else "Unknown",
                            axis=1
                        )
                        df_merged = df_merged[df_merged["gender"] != "Unknown"]


                        df_merged = df_merged[["image", "gender", "age", "race",
                                               "gender_fair", "gender_deepface", "gender_insightface", "equal_gender",
                                               "gender_scores_fair", "gender_scores_deep",
                                               "age_fair", "age_deepface", "age_insightface", "age_bin_fair",
                                               "age_bin_insightface", "equal_age",
                                               "age_scores_fair", "age_diff",
                                               "race_fair", "race4_deepface", "equal_race",
                                               "face_fair", "bbox"]]

                        # remove detected faces that are duplicates with equal bbox
                        df_merged["bbox"] = df_merged["bbox"].apply(parse_bbox)  # parse string to list
                        df_merged = deduplicate_faces(df_merged)
                        # keep only images with exactly one face
                        counts = df_merged["image"].value_counts()
                        valid_images = counts[counts == 1].index
                        df_merged = df_merged[df_merged["image"].isin(valid_images)].copy()

                        agree_rate = df_merged["equal_gender"].sum()
                        disagree_rate = len(df_merged) - agree_rate
                        count.append(disagree_rate)
                        print(f"Disagreement gender: {disagree_rate}")

                        agree_rate = df_merged["equal_age"].sum()
                        disagree_rate = len(df_merged) - agree_rate
                        count_age.append(disagree_rate)
                        print(f"Disagreement age: {disagree_rate}")

                        agree_rate = df_merged["equal_race"].sum()
                        disagree_rate = len(df_merged) - agree_rate
                        count_race.append(disagree_rate)
                        print(f"Disagreement race: {disagree_rate}")

                        # Save updated CSV
                        df_merged.to_csv(os.path.join(output_specialty_path, group + ".csv"), index=False)

    print("gender:", sum(count)/len(count), sum(count))
    print("age:", sum(count_age)/len(count), sum(count_age))
    print("race:", sum(count_race)/len(count), sum(count_race))


def postprocess_demographics(base_folder, output_folder, exp):
    # input_folder = "results/analysis_results_raw/"  # folder containing model subfolders
    # output_folder = "results/analysis_results_merged/"
    # extract_demographics(base_folder, output_folder)
    calc_group_percentage(exp)
    fill_missing_groups()
