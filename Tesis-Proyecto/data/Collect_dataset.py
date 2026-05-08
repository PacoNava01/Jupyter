import cv2
import numpy as np
import csv
from picamera2 import Picamera2
import time

OUTPUT_FILE = "Jupyter/Tesis-Proyecto/data/dataset.csv"

# HSV rango rojo
low_red1 = np.array([0, 120, 80])
up_red1  = np.array([10, 255, 255])

low_red2 = np.array([170, 120, 80])
up_red2  = np.array([180, 255, 255])

current_label = 1  # 1=objeto, 0=fondo
dataset = []

last_contours = []
last_hsv = None


# --- FUNCIONES DE EXTRACCIÓN ---
def extract_features(contour, hsv_frame):

    area = cv2.contourArea(contour)

    if area < 10:
        return None

    # Geometría
    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = w / float(h)

    perimeter = cv2.arcLength(contour, True)

    circularity = (
        (4 * np.pi * area) / (perimeter ** 2)
        if perimeter > 0 else 0
    )

    # Solidez
    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull)

    solidity = (
        float(area) / hull_area
        if hull_area > 0 else 0
    )

    # Color promedio
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

    cv2.drawContours(mask, [contour], -1, 255, -1)

    mean_val = cv2.mean(hsv_frame, mask=mask)

    return [
        area,
        aspect_ratio,
        circularity,
        solidity,
        mean_val[0],
        mean_val[1],
        mean_val[2]
    ]


# --- EVENTO DE MOUSE ---
def mouse_callback(event, x, y, flag, param):

    global dataset
    global current_label
    global last_contours
    global last_hsv

    if event == cv2.EVENT_LBUTTONDOWN:

        for c in last_contours:

            if cv2.pointPolygonTest(c, (x, y), False) >= 0:

                features = extract_features(c, last_hsv)

                if features:

                    dataset.append(
                        features + [current_label]
                    )

                    print(
                        f"[+] {current_label} guardado. "
                        f"Total {len(dataset)}"
                    )


def init_cam():

    try:

        cam = Picamera2()

        config = cam.create_video_configuration(
            main={
                "size": (640, 480),
                "format": "RGB888"
            }
        )

        cam.configure(config)

        cam.start()

        print("Cámara inicializada correctamente.")

        return cam

    except Exception as e:

        print(f"Error al inicializar la cámara: {e}")

        return None


if __name__ == "__main__":

    cam = init_cam()

    if cam is None:
        exit()

    cv2.namedWindow("Frame")

    cv2.setMouseCallback(
        "Frame",
        mouse_callback
    )

    try:

        while True:

            frame = cam.capture_array()

            frame = cv2.rotate(
                frame,
                cv2.ROTATE_180
            )

            hsv = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2HSV
            )

            # Máscara roja
            m1 = cv2.inRange(
                hsv,
                low_red1,
                up_red1
            )

            m2 = cv2.inRange(
                hsv,
                low_red2,
                up_red2
            )

            mask = cv2.add(m1, m2)

            # Limpieza
            kernel = np.ones((5, 5), np.uint8)

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel
            )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            last_contours = contours
            last_hsv = hsv

            display = frame.copy()

            cv2.drawContours(
                display,
                [
                    c for c in contours
                    if cv2.contourArea(c) > 400
                ],
                -1,
                (0, 255, 0),
                1
            )

            status_txt = (
                f"MODO: "
                f"{'OBJETO' if current_label == 1 else 'FONDO'} "
                f"| Muestras: {len(dataset)}"
            )

            cv2.putText(
                display,
                status_txt,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("Frame", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('o'):
                current_label = 1

            elif key == ord('f'):
                current_label = 0

            elif key == ord('s'):

                with open(
                    OUTPUT_FILE,
                    "w",
                    newline=""
                ) as f:

                    writer = csv.writer(f)

                    writer.writerow([
                        "area",
                        "aspect_ratio",
                        "circularity",
                        "solidity",
                        "h",
                        "s",
                        "v",
                        "label"
                    ])

                    writer.writerows(dataset)

                print("Dataset guardado")

            elif key == 13:
                break

    finally:

        cam.stop()

        cv2.destroyAllWindows()