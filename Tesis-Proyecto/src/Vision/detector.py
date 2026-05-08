import cv2
import numpy as np
import joblib


class ObjectDetector:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        #Precalculamos el kernel para operaciones morfologicas
        self.Kernel = np.ones((5,5),np.uint8)

    def extract_features(self, contour,h_channel,s_channel,v_channel):
        '''
        Extrae caracterizticas optimizadas sin crear mascaras nuevas
        '''
        area = cv2.contourArea(contour)
        if area == 0: return None

        #1. Geometria (es invariante al tamaño absoluto hasta cierto punto)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        #Solidez: Area/Area del convex Hull (detectamos si es concavo o convexo)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(area)/hull_area
        else:
            solidity = 0
        '''
        cerca de 1 → muy sólida/convexa
        lejos de 1 → irregular o con huecos internos
        '''

        #Circularidad
        perimeter = cv2.arcLength(contour, True)
        circularity = 0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else: 
            circularity = 0

        #2. Color

        '''
        en lugar de cv2.mean con mascara, usamos una aproximacion
        rapida basada en el bounding box para reducir el ROI
        '''
        roi_h = h_channel[y:y+h,x:x+w]
        h_mean = np.mean(roi_h)
        s_mean = np.mean(s_channel[y:y+h,x:x+w])
        v_mean = np.mean(v_channel[y:y+h,x:x+w])

        return np.array([area,aspect_ratio,circularity,
                         solidity,h_mean,s_mean,v_mean],
                         dtype=np.float32)
    

    def process_frame(self, frame_hsv, hsv_mask):
        '''
        pipeline optimizado recibiendo el frame hsv de entrada
        '''
        h_ch, s_ch, v_ch = cv2.split(frame_hsv)
        
        contornos,_ = cv2.findContours(hsv_mask,cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        output_mask = np.zeros_like(hsv_mask)

        #Lista para predicciones en batch (opcional, pero mejora velocidad de ML)
        candidates_features = []
        candidates_contours = []

        for c in contornos:
            if cv2.contourArea(c) < 550:
                continue
            feat = self.extract_features(c,h_ch,s_ch,v_ch)
            if feat is not None:
                candidates_features.append(feat)
                candidates_contours.append(c)
            
            if len(candidates_features) > 0:
                #Prediccion masiva (màs rapido que uno por uno)
                predictions = self.model.predict(candidates_features)

                for i, is_target in enumerate(predictions):
                    if is_target == 1:
                        cv2.drawnContours(output_mask,[candidates_contours[i]],-1,255,-1)
            
            return output_mask
    

#  PRE-FILTRO HSV (un poco màs lijgero)
def obtener_mask(frame_hsv, low_hsv, up_hsv):
    mask = cv2.inRange(frame_hsv, low_hsv, up_hsv)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) #Elimina ruido pequeño
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) #Cierra huecos
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask
