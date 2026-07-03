import cv2
import numpy as np
import joblib

class ObjectDetector:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.kernel = np.ones((5,5), np.uint8)

    def extract_features(self, contour, h_channel, s_channel, v_channel):
        area = cv2.contourArea(contour)
        if area == 0: return None

        # 1. Geometría
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # 2. Color (ROI segura)
        mask_roi = np.zeros_like(h_channel)
        #cv2.drawnContours(mask_roi,[contour],-1,255,-1)

        h_mean = cv2.mean(h_channel,mask_roi)[0]
        s_mean = cv2.mean(s_channel,mask_roi)[1]
        v_mean = cv2.mean(v_channel,mask_roi)[2]
        
        '''
        roi_h = h_channel[y:y+h, x:x+w]
        roi_s = s_channel[y:y+h, x:x+w]
        roi_v = v_channel[y:y+h, x:x+w]
        
        h_mean = np.mean(roi_h)
        s_mean = np.mean(roi_s)
        v_mean = np.mean(roi_v)
        '''

        return np.array([area, aspect_ratio, circularity, solidity, h_mean, s_mean, v_mean], dtype=np.float32)

    def process_frame(self, frame_hsv, hsv_mask):
        #Asegurar que se dividan los cnales HSV
        h_ch, s_ch, v_ch = cv2.split(frame_hsv)
        
        contornos, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output_mask = np.zeros_like(hsv_mask)
        candidates_features = []
        candidates_contours = []

        for c in contornos:
            if cv2.contourArea(c) < 400:
                continue
            
            feat = self.extract_features(c, h_ch, s_ch, v_ch)
            if feat is not None:
                candidates_features.append(feat)
                candidates_contours.append(c)
        
        # ---  Indentación y Escalado ---
        if len(candidates_features) > 0:
            # Primero escalamos todo el batch, luego predecimos
            features_scaled = self.scaler.transform(candidates_features)
            predictions = self.model.predict(features_scaled)

            for i, is_target in enumerate(predictions):
                if is_target == 1:
                    cv2.drawContours(output_mask, [candidates_contours[i]], -1, 255, -1)
        
        return output_mask

def obtener_mask(frame_hsv, low_hsv, up_hsv):
    mask = cv2.inRange(frame_hsv, low_hsv, up_hsv)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
    # GaussianBlur es opcional pero ayuda a suavizar bordes para contornos
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask
