# classification/analyzers/deepface_analyzer.py

from deepface import DeepFace


class DeepFaceAnalyzer:

    def predict(self, img_path, img_file):
        result = DeepFace.analyze(
            img_path,
            actions=["age", "gender", "race", "emotion"],
            enforce_detection=False,
            detector_backend="retinaface"
        )[0]

        row = {
            "image": img_file,
            "age_deepface": result.get("age"),
            "gender_deepface": result.get("dominant_gender"),
            "race_deepface": result.get("dominant_race"),
            "emotion_deepface": result.get("dominant_emotion"),
            "face_confidence_deepface": result.get("face_confidence"),
        }

        # flatten distributions
        for k, v in result.get("gender", {}).items():
            row[f"gender_{k}"] = v

        for k, v in result.get("race", {}).items():
            row[f"race_{k}"] = v

        for k, v in result.get("emotion", {}).items():
            row[f"emotion_{k}"] = v

        return row