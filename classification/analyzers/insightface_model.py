# classification/analyzers/insightface_analyzer.py

import cv2
from insightface.app import FaceAnalysis


class InsightFaceAnalyzer:

    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0)

    def predict(self, img_path, img_file):
        img = cv2.imread(img_path)
        faces = self.app.get(img)

        if len(faces) == 0:
            img = cv2.resize(img, None, fx=2, fy=2)
            faces = self.app.get(img)

        row = {"image": img_file}

        if len(faces) > 0:
            face = max(
                faces,
                key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1])
            )
            row["age_insightface"] = face.age
            row["gender_insightface"] = face.gender

        return row