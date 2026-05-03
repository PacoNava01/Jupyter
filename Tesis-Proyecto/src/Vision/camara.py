import cv2
import numpy as np
from picamera2 import Picamera2
import time

'''
Script enfocado unicamente en la captura de frames
'''
def init_cam():
    try:
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"})
        cam.configure(config)
        cam.start()
        print("Cámara inicializada correctamente.")
        return cam #Regresamos el objeto camara
    
    except Exception as e:
        print(f"Error al inicializar la cámara: {e}")
        return None

if __name__ == "__main__":    

    try:
        camara = init_cam() #Instanciamos el objeto camara
        if camara is None: exit()

        cv2.namedWindow("Frame capturado")
        while True:
        
            frame_raw = camara.capture_array()

            if frame_raw is None: break
            frame_raw = cv2.rotate(frame_raw,cv2.ROTATE_180)

            cv2.imshow("Frame capturado",frame_raw)

            if cv2.waitKey(1) & 0xFF == 13: # Enter para salir
                break

    finally:
        if 'camara' in locals():
            camara.stop()
        cv2.destroyAllWindows()    