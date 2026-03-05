#Librerias
import numpy as np
import cv2
import os

#Librerias propias
from Vision import *






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
            frame,centroide,w_bbox = obtener_contornos(mask,frame_bgr.copy(),min_area=650)
            resultado = aplicar_mascara(cv2.cvtColor(frame_bgr_blur, cv2.COLOR_BGR2RGB), mask)

            if centroide:
                cv2.putText(frame,f"crentroide pos: {centroide}",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.circle(frame, centro_cam, 5, (255, 0, 0), -1) # Centro de la cámara
            
            #--- Calibraciòn --- objeto de 15cm2 a 50 cm
            Distancia = calcular_distancia(15,w_bbox,focal_px=770 ) # Focal length ajustada empíricamente
            print(f"Distancia estimada: {Distancia:.2f} cm")
            
           


            
            mostrar_frame("Camara", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #Mostramos en RGB para que se vea el color correcto
            mostrar_frame("Mascara", mask)
            mostrar_frame("Resultados", resultado)

            if cv2.waitKey(1) & 0xFF == 13:
                break
    finally:
        camara.stop()
        cv2.destroyAllWindows()
