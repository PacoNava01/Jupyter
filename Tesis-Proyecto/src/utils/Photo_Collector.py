import cv2
from picamera2 import Picamera2
import time
import os

'''
Capturamos una foto cada x segundos mientras se "graba un video"
'''

# Parametros
intervalo = 3
nombre_ventana = "Frame"
contador = 1

# Directorio destino
DICT_DESTINO = "/home/pacon/Tesis_pacon/Jupyter/Tesis-Proyecto/data/ArUcocalib"

# Asegurar que el directorio exista para evitar errores
os.makedirs(DICT_DESTINO, exist_ok=True)

ultimo_chek = time.time()

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


if __name__ == '__main__':
    try:
        camara = init_cam()
        if camara is None: 
            exit()

        cv2.namedWindow(nombre_ventana)
        
        while True:
            frame = camara.capture_array() 
            frame = cv2.rotate(frame, cv2.ROTATE_180) # frame para mostrar
            
            # Clonamos el frame original ANTES de pintarle texto para guardarlo limpio
            frame2save = frame.copy() 
            
            ahora = time.time()

            if ahora - ultimo_chek >= intervalo:
                # Nombre del archivo con la ruta completa
                nombre_archivo = f"frame_calib_{contador}.jpg"
                ruta_completa = os.path.join(DICT_DESTINO, nombre_archivo)
                
                # Guardamos la imagen
                cv2.imwrite(ruta_completa, frame2save)
                print(f"Foto guardada: {ruta_completa}")

                ultimo_chek = ahora
                contador += 1

            # Mostramos en pantalla el número de la PRÓXIMA foto que se va a tomar
            cv2.putText(frame, f"Prox. Foto: #{contador}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(nombre_ventana, frame)
            
            # Detecta la tecla Enter (13) para salir
            key = cv2.waitKey(1) & 0xFF
            if key == 13: 
                break

    finally:
        if 'camara' in locals() and camara is not None:
            camara.stop()
        cv2.destroyAllWindows()