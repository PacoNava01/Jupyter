#Librerias
import numpy as np
import cv2
import os
import time 
import csv

#Librerias propias
from Vision import *
Nombre_log = "Caracterizacion_polarizadores.csv"
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

    areas = [] # Lista para almacenar las áreas de los objetos detectados
    print("Presiona 's' para capturar datos del polarizador\nPresiona 'Enter' " \
    "para terminar el programa")
    try:
        while True:
            frame_bgr = capture_frame(camara)
            if frame_bgr is None: break
            alto, ancho, _ = frame_bgr.shape
            centro_cam = (ancho//2, alto//2)

            # --- Procesamiento ---
            # Aplicamos un filtro de desenfoque para reducir ruido
            frame_bgr_blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
            mask = detectar_color(frame_bgr_blur,(0,0,255), thres_color)
            frame,centroide,w_bbox,h_bbox = obtener_contornos(mask,frame_bgr.copy(),min_area=650)
            resultado = aplicar_mascara(cv2.cvtColor(frame_bgr_blur, cv2.COLOR_BGR2RGB), mask)
            
            #Almacenar los datos del area de la caja y la distancia 
            area_bbox = w_bbox * h_bbox
            areas.append(area_bbox)
            
           

            if time.time() - ultim_check > intervalo: #Cada 5 seundos mostramos el promedio de las mediciones
                if w_bbox and h_bbox:
                    area_promedio = sum(areas) / len(areas)
                    ultim_check = time.time()
                    #print(f"Area del objeto detectado: {area_promedio:.2f} px2")
                    #print(f"Distancia estimada: {calcular_distancia(15,w_bbox, focal_px=973):.2f} cm") # Focal length ajustada empíricamente objeto de 15cm2 a 50 cm
                    areas = [] # Reiniciamos la lista para el próximo intervalo
            
            #--- Visualizacion de datos en pantalla ---
            if centroide:
                cv2.putText(frame,f"crentroide pos: {centroide}",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.circle(frame, centro_cam, 5, (255, 0, 0), -1) # Centro de la cámara
            
            #--- Calibraciòn --- objeto de 15cm2 a 50 cm
        
            mostrar_frame("Camara", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #Mostramos en RGB para que se vea el color correcto
            mostrar_frame("Mascara", mask)
            mostrar_frame("Resultados", resultado)
            
            #--- Logica de teclas---
            key = cv2.waitKey(1) & 0xFF

            #Si presionas 's' Se detiene un momento el programa y guarda
            if key == ord('s'):
                analizar_metricas_polarizacion(frame_bgr,mask,resultado,Nombre_log)
            
            #si presionas Enter (13) nos salimos
            if key == 13:
                break
    finally:
        camara.stop()
        cv2.destroyAllWindows()
