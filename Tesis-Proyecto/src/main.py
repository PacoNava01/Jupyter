from Vision.detector import ObjectDetector,obtener_mask
from picamera2 import Picamera2
import cv2
import numpy as np


# ----------------------------
# INIT
# ----------------------------
detector = ObjectDetector("Jupyter/Tesis-Proyecto/data/model.pkl")

cam = Picamera2()
config = cam.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
cam.configure(config)
cam.start()

# HSV rojo (igual que dataset)
low_red1 = np.array([0, 120, 80])
up_red1  = np.array([10, 255, 255])
low_red2 = np.array([170, 120, 80])
up_red2  = np.array([180, 255, 255])

print("Iniciando prueba en tiempo real...")

# ----------------------------
# LOOP
# ----------------------------
while True:
    frame = cam.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # 1. máscara HSV base
    mask = obtener_mask(frame, low_red1, up_red1)
    mask2 = obtener_mask(frame, low_red2, up_red2)
    hsv_mask = cv2.add(mask, mask2)

    # 2. ML refine
    final_mask = detector.process_frame(frame, hsv_mask)

    # 3. visualizar
    cv2.imshow("Original", frame)
    cv2.imshow("HSV Mask", hsv_mask)
    cv2.imshow("ML Mask", final_mask)

    if cv2.waitKey(1) & 0xFF == 13:
        break

cam.stop()
cv2.destroyAllWindows()
    