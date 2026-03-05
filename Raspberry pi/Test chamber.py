#Librerias
import numpy as np
import cv2
import os
import time 

#Librerias propias
from Vision import *



ultim_check = time.time()
intervalo = 5 # segundos


#--- Playground ---
if __name__ == "__main__":
    camara = init_camara()
    if camara is None: exit()
    os.system("clear")

    cv2.namedWindow("Camara",cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Mascara",cv2.WINDOW_AUTOSIZE)  
    cv2.namedWindow("Resultados",cv2.WINDOW_AUTOSIZE)

    
    try:
        while True:
            frame_bgr = capture_frame(camara)
            if frame_bgr is None: break
            alto, ancho, _ = frame_bgr.shape

            centro_cam = (ancho//2, alto//2)
            # Aplicamos un filtro de desenfoque para reducir ruido
            frame_bgr_blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)

            mask = detectar_color(frame_bgr_blur,(0,0,255), thres_color)
            frame,centroide,w_bbox,h_bbox = obtener_contornos(mask,frame_bgr.copy(),min_area=650)
            resultado = aplicar_mascara(cv2.cvtColor(frame_bgr_blur, cv2.COLOR_BGR2RGB), mask)

            if time.time() - ultim_check > intervalo:
                if w_bbox and h_bbox:
                    ultim_check = time.time()
                    print(f"Area del objeto detectado: {w_bbox*h_bbox} px2")
                    print(f"Distancia estimada: {calcular_distancia(15,w_bbox, focal_px=973):.2f} cm") # Focal length ajustada empíricamente objeto de 15cm2 a 50 cm

            if centroide:
                cv2.putText(frame,f"crentroide pos: {centroide}",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.circle(frame, centro_cam, 5, (255, 0, 0), -1) # Centro de la cámara
            
            #--- Calibraciòn --- objeto de 15cm2 a 50 cm
            
            
           


            
            mostrar_frame("Camara", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #Mostramos en RGB para que se vea el color correcto
            mostrar_frame("Mascara", mask)
            mostrar_frame("Resultados", resultado)

            if cv2.waitKey(1) & 0xFF == 13:
                break
    finally:
        camara.stop()
        cv2.destroyAllWindows()
