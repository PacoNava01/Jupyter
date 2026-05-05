from Vision.detector import ObjectDetector, obtener_mask
from Vision.camara import init_cam
from Hardware.Servomotores.MG996R import init_servos, Servo2Pos

import cv2
import numpy as np


# Modelo (opcional, realmente actúa como filtro sobre HSV)
detector = ObjectDetector("Jupyter/Tesis-Proyecto/data/model.pkl")

# Cámara
cam = init_cam()
if cam is None:
    exit()

# Servo (solo eje X)
servo_x, _ = init_servos()
servo_x_angle = 80  # posición inicial

# Centro del frame (referencia)
frame_center_x = 320

# ----------------------------
# TRACKING SIMPLE (SIN TRACKER)
# ----------------------------
# Guarda el último objeto seguido
prev_target = None

# ----------------------------
# SUAVIZADO (EMA)
# ----------------------------
alpha = 0.4  # menor = más reactivo

# ----------------------------
# PID CONFIG
# ----------------------------
# Kp: responde al error actual 
# Ki: corrige error acumulado (offset) 
# Kd: reduce oscilaciones (suaviza cambios bruscos)
Kp = 10
Ki = 0.0   # desactivado por estabilidad
Kd = 0.1

integral_x = 0
prev_error_x = 0

# Zona muerta (en error normalizado)
dead_zone = 0.05

# ----------------------------
# HSV rojo
# ----------------------------
low_red1 = np.array([0, 120, 90])
up_red1  = np.array([10, 255, 255])

low_red2 = np.array([170, 120, 90])
up_red2  = np.array([180, 255, 255])

print("Iniciando prueba en tiempo real...")
Servo2Pos(servo_x, 80)

# ----------------------------
# LOOP PRINCIPAL
# ----------------------------
while True:

    # Captura y orientación
    frame = cam.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    display_frame = frame.copy()

    # ----------------------------
    # 1. DETECCIÓN POR COLOR (HSV)
    # ----------------------------
    mask1 = obtener_mask(frame, low_red1, up_red1)
    mask2 = obtener_mask(frame, low_red2, up_red2)
    hsv_mask = cv2.add(mask1, mask2)

    # ----------------------------
    # 2. FILTRADO (pseudo ML)
    # ----------------------------
    final_mask = detector.process_frame(frame, hsv_mask)

    # ----------------------------
    # 3. CONTORNOS → CANDIDATOS
    # ----------------------------
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)

        # Filtrar ruido
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

        # Primer frame → elegir el más grande
        if prev_target is None:
            best_centroid = max(candidates, key=lambda x: x[2])[:2]

        else:
            # Elegir el más cercano al anterior (continuidad)
            min_dist = float("inf")

            for (cx, cy, area) in candidates:
                dist = abs(cx - prev_target[0])  # solo eje X

                if dist < min_dist:
                    min_dist = dist
                    best_centroid = (cx, cy)

    # ----------------------------
    # 5. CONTROL (SI HAY OBJETO)
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

        # -------- ERROR NORMALIZADO --------
        error_x = (smooth_x - frame_center_x) / frame_center_x

        # Zona muerta (evita jitter)
        if abs(error_x) < dead_zone:
            error_x = 0

        # -------- PID (P + D) --------
        integral_x += error_x
        derivative_x = error_x - prev_error_x

        control = (Kp * error_x) + (Kd * derivative_x)
        prev_error_x = error_x

        # -------- ÁNGULO OBJETIVO --------
        target_angle = 80 - control

        # Limitar rango físico
        target_angle = max(20, min(140, target_angle))

        # -------- LIMITADOR DE VELOCIDAD --------
        # Evita movimientos bruscos
        max_step = 2  # grados por frame

        delta = target_angle - servo_x_angle
        delta = max(-max_step, min(max_step, delta))

        servo_x_angle += delta

        # Mover servo
        Servo2Pos(servo_x, servo_x_angle)

        # Dibujar objetivo activo
        cv2.circle(display_frame, (smooth_x, smooth_y), 8, (0, 255, 0), 2)

    # ----------------------------
    # DEBUG VISUAL
    # ----------------------------
    # Dibujar todos los candidatos detectados
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