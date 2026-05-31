import cv2
from picamera2 import Picamera2
import time

"""
Captura una foto 5 segundos después
de hacer click izquierdo
"""

# Variables globales
temporizador_activo = False
tiempo_click = 0

TIEMPO_ESPERA = 3


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


# Evento del mouse
def click_mouse(event, x, y, flags, param):

    global temporizador_activo
    global tiempo_click

    # Click izquierdo
    if event == cv2.EVENT_LBUTTONDOWN:

        # Guardar instante del click
        tiempo_click = time.time()

        temporizador_activo = True

        print("Temporizador iniciado...")


if __name__ == "__main__":

    try:

        camara = init_cam()

        if camara is None:
            exit()

        nombre_ventana = "Frame capturado"

        cv2.namedWindow(nombre_ventana)

        cv2.setMouseCallback(nombre_ventana, click_mouse)

        contador = 0

        while True:

            frame_raw = camara.capture_array()

            if frame_raw is None:
                break

            frame_raw = cv2.rotate(frame_raw, cv2.ROTATE_180)

            # Texto principal
            '''cv2.putText(
                frame_raw,
                "Click izquierdo para tomar foto",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )'''

            # Si el temporizador está activo
            if temporizador_activo:

                tiempo_actual = time.time()

                tiempo_transcurrido = tiempo_actual - tiempo_click

                tiempo_restante = int(
                    TIEMPO_ESPERA - tiempo_transcurrido
                ) + 1

                # Mostrar cuenta regresiva
                if tiempo_restante > 0:

                    cv2.putText(
                        frame_raw,
                        f"Foto en {tiempo_restante}",
                        (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 0, 255),
                        4
                    )

                # Tomar fotografía
                else:
                    ruta = "/home/pacon/Tesis_pacon/Jupyter/Tesis-Proyecto/src/Vision/ArUcocalib"
                    nombre = f"{ruta}/calib_{contador}.jpg"

                    cv2.imwrite(nombre, frame_raw)

                    print(f"Imagen guardada: {nombre}")

                    contador += 1

                    temporizador_activo = False

            # Mostrar frame
            cv2.imshow(nombre_ventana, frame_raw)

            # Enter para salir
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == 13:
                break

    finally:

        if 'camara' in locals():
            camara.stop()

        cv2.destroyAllWindows()