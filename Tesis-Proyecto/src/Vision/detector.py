import cv2
import numpy as np
import joblib


class ObjectDetector:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    # 🔹 FEATURES
    def extract_features(self, contour, frame_hsv):
        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        perimeter = cv2.arcLength(contour, True)
        circularity = 0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)

        # máscara del contorno
        mask = np.zeros(frame_hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        mean_val = cv2.mean(frame_hsv, mask=mask)
        h_mean, s_mean, v_mean = mean_val[:3]

        return np.array([
            area, aspect_ratio, circularity,
            h_mean, s_mean, v_mean
        ], dtype=np.float32)

    # 🔹 CLASIFICADOR
    def classify_contour(self, contour, frame_hsv):
        features = self.extract_features(contour, frame_hsv)
        prediction = self.model.predict([features])
        return prediction[0] == 1

    # 🔹 PIPELINE PRINCIPAL
    def process_frame(self, frame_bgr, hsv_mask):
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        contornos, _ = cv2.findContours(
            hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        output_mask = np.zeros_like(hsv_mask)

        for c in contornos:
            if cv2.contourArea(c) < 500:
                continue

            if self.classify_contour(c, frame_hsv):
                cv2.drawContours(output_mask, [c], -1, 255, -1)

        return output_mask


# 🔹 PRE-FILTRO HSV (se queda fuera de la clase)
def obtener_mask(frame_bgr, low_hsv, up_hsv):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_hsv, up_hsv)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask