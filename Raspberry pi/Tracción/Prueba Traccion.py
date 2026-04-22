# --- Librerias propias---
from Desplazamiento import Carro

# --- Librerias ---
import numpy as np
import cv2
import time
import os

if __name__ == "__main__":
    #Pines: (IN1,IN2,ENB) para cada lado
    pines_izq = (17,27,12)
    pines_der = (23,22,13)
    pin_stby = 24

    carrito = None
    
    cv2.namedWindow("Control Carrito")

    try:
        carrito = Carro(pines_izq,pines_der)
        print("Robot activado...")
        moviendose = False
        intervalo = 2
        ultim_check = time.time()

        while True:
            tecla = cv2.waitKey(1) & 0xFF
            tiempo_actual = time.time()


            # --- TEMPORIZADOR DE MEDICIÓN ---
            if moviendose and (tiempo_actual - ultim_check) > intervalo:
                carrito.detener()
                print("Intervalo cumplido: Deteniendo para medición...")
                moviendose = False
                impulso_activo = False
            
            # --- CONTROL DE TECLADO ---
            if tecla == ord('w'):
                print("Avanzando...")
                carrito.mover(0.5,0.5)
                ultim_check = tiempo_actual
                moviendose = True
                

            elif tecla == ord('s'):
                print("Retrocediendo...")
                carrito.mover(-1,-1)
                ultim_check = tiempo_actual
                moviendose = True
            
            elif tecla == ord(' '):
                carrito.detener()
                moviendose = False
                impulso_activo = False
                print("Parada de emergencia")

            elif tecla == 27:  # ESC
                break
                

    except KeyboardInterrupt:
        if carrito is not None:
            carrito.detener()
            carrito.apagar_driver()
            print("\nPrograma terminado y driver en Standby")
    
    except Exception as e:
        print(f"Ocurrió un error: {e}")