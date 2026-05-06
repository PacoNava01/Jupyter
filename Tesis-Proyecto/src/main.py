from Vision.detector import ObjectDetector, obtener_mask
from Vision.camara import init_cam
from Hardware.Servomotores.MG996R import init_servos, Servo2Pos

import cv2
import numpy as np


# ----------------------------
# INIT
# ----------------------------

detector = ObjectDetector("Jupyter/Tesis-Proyecto/data/model.pkl")

cam = init_cam()
if cam is None:
    exit()

# Servo (solo eje X)
servo_x, _ = init_servos()
servo_x_angle = 90  # centro inicial

# Centro del frame
frame_center_x = 320

# Última posición del objeto (para continuidad)
prev_target = None

# Suavizado (EMA)
alpha = 0.5

# ----------------------------
# CONTROL PROPORCIONAL
# ----------------------------
# Relación pixel → grados ( parámetro más importante)
gain = 0.12

# Límite de velocidad del servo (evita saltos bruscos)
max_step = 2

# Zona muerta en pixeles (evita vibración)
dead_zone_px = 10

# ----------------------------
# HSV rojo
# ----------------------------
low_red1 = np.array([0, 120, 90])
up_red1  = np.array([10, 255, 255])

low_red2 = np.array([170, 120, 90])
up_red2  = np.array([180, 255, 255])

print("Iniciando prueba en tiempo real...")
Servo2Pos(servo_x, servo_x_angle)


# ----------------------------
# LOOP PRINCIPAL
# ----------------------------
while True:

    frame = cam.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    display_frame = frame.copy()

    # ----------------------------
    # 1. DETECCIÓN HSV
    # ----------------------------
    mask1 = obtener_mask(frame, low_red1, up_red1)
    mask2 = obtener_mask(frame, low_red2, up_red2)
    hsv_mask = cv2.add(mask1, mask2)

    # ----------------------------
    # 2. FILTRADO
    # ----------------------------
    final_mask = detector.process_frame(frame, hsv_mask)

    # ----------------------------
    # 3. CONTORNOS
    # ----------------------------
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < 550:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        candidates.append((cx, cy, area))

    # ----------------------------
    # 4. SELECCIÓN DE OBJETIVO
    # ----------------------------
    best_centroid = None

    if len(candidates) > 0:

        # usar siempre el más grande
        largest = max(candidates, key=lambda x: x[2])
        largest_centroid = largest[:2]

        if prev_target is None:
            best_centroid = largest_centroid
        else:
            # si no "salta", seguir el grande
            dist = abs(largest_centroid[0] - prev_target[0])

            if dist < 120:
                best_centroid = largest_centroid
            else:
                # transición suave
                min_dist = float("inf")
                for (cx, cy, area) in candidates:
                    d = abs(cx - prev_target[0])
                    if d < min_dist:
                        min_dist = d
                        best_centroid = (cx, cy)

    # ----------------------------
    # 5. CONTROL (SIN PID)
    # ----------------------------
    if best_centroid is not None:

        cx, cy = best_centroid

        # -------- SUAVIZADO --------
        if prev_target is not None:
            smooth_x = int(alpha * prev_target[0] + (1 - alpha) * cx)
            smooth_y = int(alpha * prev_target[1] + (1 - alpha) * cy)
        else:
            smooth_x, smooth_y = cx, cy

        prev_target = (smooth_x, smooth_y)

        # -------- ERROR EN PIXELES --------
        error_px = smooth_x - frame_center_x

        # zona muerta
        if abs(error_px) < dead_zone_px:
            error_px = 0

        # -------- CONTROL DIRECTO --------
        target_angle = 90 - error_px * gain

        # limitar rango físico
        target_angle = max(10, min(170, target_angle))

        # -------- LIMITADOR DE VELOCIDAD --------
        delta = target_angle - servo_x_angle
        delta = max(-max_step, min(max_step, delta))

        servo_x_angle += delta

        # mover servo
        Servo2Pos(servo_x, servo_x_angle)

        # dibujar objetivo activo
        cv2.circle(display_frame, (smooth_x, smooth_y), 8, (0, 255, 0), 2)

    # ----------------------------
    # DEBUG
    # ----------------------------
    for (cx, cy, area) in candidates:
        cv2.circle(display_frame, (cx, cy), 4, (0, 0, 255), -1)

    # ----------------------------
    # VISUALIZACIÓN
    # ----------------------------
    ML_mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((display_frame, ML_mask_bgr))

    cv2.imshow("Comparacion", combined)

    if cv2.waitKey(1) & 0xFF == 13:
        break


# ----------------------------
# CIERRE
# ----------------------------
cam.stop()
cv2.destroyAllWindows()