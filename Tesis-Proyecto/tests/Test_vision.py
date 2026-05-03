import cv2
import numpy as np
from picamera2 import Picamera2

# =========================================================
# CAMARA
# =========================================================
def init_cam():
    try:
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        print("Cámara inicializada correctamente.")
        return cam
    except Exception as e:
        print(f"Error al inicializar la cámara: {e}")
        return None


# =========================================================
# SEGMENTACION HSV
# =========================================================
def obtener_mask(frame_rgb, low_hsv, up_hsv):
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_hsv, up_hsv)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# =========================================================
# CONTORNOS → CENTROIDES (LISTO PARA TRACKING)
# =========================================================
def procesar_contornos(mask, frame_para_dibujar, min_area=350):

    contornos, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centroides_validos = []

    for c in contornos:

        area = cv2.contourArea(c)
        if area < min_area:
            continue

        # Bounding box
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)

        # Circularidad
        perimeter = cv2.arcLength(c, True)
        circularidad = 0
        if perimeter > 0:
            circularidad = 4 * np.pi * area / (perimeter ** 2)

        # FILTROS GEOMÉTRICOS
        if not (0.3 < aspect_ratio < 3.0):
            continue

        if circularidad < 0.3:
            continue

        # CENTROIDE
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        centroides_validos.append((cx, cy))

        # VISUALIZACIÓN
        cv2.drawContours(frame_para_dibujar, [c], -1, (0, 255, 0), 2)
        cv2.circle(frame_para_dibujar, (cx, cy), 5, (0, 0, 255), -1)

    return frame_para_dibujar, centroides_validos


# =========================================================
# RANGOS HSV ROJO
# =========================================================
low_red1 = np.array([0, 100, 100], dtype=np.uint8)
up_red1 = np.array([10, 255, 255], dtype=np.uint8)
low_red2 = np.array([170, 100, 100], dtype=np.uint8)
up_red2 = np.array([180, 255, 255], dtype=np.uint8)


# =========================================================
# MAIN LOOP
# =========================================================
try:
    camara = init_cam()
    if camara is None:
        exit()

    cv2.namedWindow("Deteccion")
    cv2.namedWindow("Mascara")

    while True:

        frame_raw = camara.capture_array()
        if frame_raw is None:
            break

        frame_raw = cv2.rotate(frame_raw, cv2.ROTATE_180)

        # -------------------------
        # SEGMENTACION
        # -------------------------
        mask1 = obtener_mask(frame_raw, low_red1, up_red1)
        mask2 = obtener_mask(frame_raw, low_red2, up_red2)
        mask_red = cv2.add(mask1, mask2)

        # -------------------------
        # DETECCION
        # -------------------------
        display_frame = frame_raw.copy()

        display_frame, centroids = procesar_contornos(
            mask_red,
            display_frame,
            min_area=350
        )

        # -------------------------
        # OUTPUT VISUAL
        # -------------------------
        for (cx, cy) in centroids:
            cv2.putText(
                display_frame,
                "OBJ",
                (cx + 5, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        # -------------------------
        # SHOW
        # -------------------------
        cv2.imshow("Deteccion", display_frame)
        cv2.imshow("Mascara", mask_red)

        if cv2.waitKey(1) & 0xFF == 13:
            break

finally:
    if 'camara' in locals():
        camara.stop()
    cv2.destroyAllWindows()