import cv2
import numpy as np
import csv
from picamera2 import Picamera2
import time



# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_FILE = "Jupyter/Tesis-Proyecto/data/dataset.csv"

# HSV rango rojo
low_red1 = np.array([0, 120, 80])
up_red1  = np.array([10, 255, 255])
low_red2 = np.array([170, 120, 80])
up_red2  = np.array([180, 255, 255])

current_label = 1  # 1 = objeto, 0 = fondo

# -----------------------------
# INIT CAM
# -----------------------------
cam = Picamera2()
config = cam.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"})
cam.configure(config)
cam.start()

time.sleep(1)

# -----------------------------
# FUNCIONES
# -----------------------------

def obtener_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, low_red1, up_red1)
    mask2 = cv2.inRange(hsv, low_red2, up_red2)
    mask = cv2.add(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask, hsv


def extract_features(contour, hsv):
    area = cv2.contourArea(contour)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)

    perimeter = cv2.arcLength(contour, True)
    circularity = 0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    mean_val = cv2.mean(hsv, mask=mask)
    h_mean, s_mean, v_mean = mean_val[:3]

    return [area, aspect_ratio, circularity, h_mean, s_mean, v_mean]


# -----------------------------
# CLICK EVENT
# -----------------------------
dataset = []

def mouse_callback(event, x, y, flags, param):
    global dataset, current_label, last_contours, last_hsv

    if event == cv2.EVENT_LBUTTONDOWN:
        for c in last_contours:
            if cv2.pointPolygonTest(c, (x, y), False) >= 0:
                features = extract_features(c, last_hsv)
                dataset.append(features + [current_label])
                print(f"[+] Guardado: {features} -> label {current_label}")
                break


cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

# -----------------------------
# LOOP
# -----------------------------
last_contours = []
last_hsv = None

import os
os.system('clear')

print("INSTRUCCIONES:")
print(" - Click izquierdo: guardar muestra")
print(" - Tecla 'o': modo OBJETO (1)")
print(" - Tecla 'f': modo FONDO (0)")
print(" - Tecla 's': guardar dataset")
print(" - ENTER: salir")





while True:
    frame = cam.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    mask, hsv = obtener_mask(frame)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # guardar para el callback
    last_contours = contours
    last_hsv = hsv

    display = frame.copy()

    for c in contours:
        if cv2.contourArea(c) > 500:
            cv2.drawContours(display, [c], -1, (0, 255, 0), 2)

    cv2.putText(display, f"Modo: {'OBJETO' if current_label == 1 else 'FONDO'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if current_label == 1 else (0, 0, 255), 2)

    cv2.imshow("Frame", display)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    #Tendremos que enseñarle al modelo que es rojo y què no lo es a partir de  200-300 muestras

    if key == ord('o'):
        current_label = 1

    elif key == ord('f'):
        current_label = 0

    elif key == ord('s'):
        with open(OUTPUT_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(dataset)
        print(f"[✔] Dataset guardado en {OUTPUT_FILE}")

    elif key == 13:  # ENTER
        break

cam.stop()
cv2.destroyAllWindows()