# --- Librerias propias---
from Desplazamiento import Carro

# --- Librerias ---
import numpy as np
import cv2
import time
from time import sleep
import os


# --- Playground

if __name__ == "__main__":
    #(forward,backward,enable)
    pines_izq = (17,27,12)
    pines_der = (23,22,13)
    stby = 24

    carrito = None
    # IMPORTANTE: OpenCV necesita una ventana para registrar teclado
    cv2.namedWindow("Control Carrito")
    tecla = None
    intervalo = 1
   
    try:
        carrito = Carro(pines_izq, pines_der, stby)
        print("Carrito instanciado. Presiona W, A, S, D. 'ESC' para salir.")
        
        moviendose = False  # <--- NUEVA BANDERA
        intervalo = 1
        ultim_check = time.time()

        while True:
            tecla = cv2.waitKey(1) & 0xFF 
            tiempo_actual = time.time()

            # --- LÓGICA DEL TEMPORIZADOR MEJORADA ---
            # Solo entra aquí si el tiempo pasó Y el carro está en movimiento
            if moviendose and (tiempo_actual - ultim_check) > intervalo:
                carrito.detener()
                print("Intervalo cumplido: Deteniendo para medición...")
                moviendose = False # <--- IMPORTANTE: Marcamos que ya se detuvo
            # ---------------------------------------

            if tecla == ord('w'):
                print("Avanzando...")
                carrito.avanzar(0.9, 0.9)
                ultim_check = time.time() 
                moviendose = True # <--- Activamos la cuenta regresiva
               
            elif tecla == ord('s'):
                print("Retrocediendo...")
                carrito.retroceder(1, 1)
                ultim_check = time.time()
                moviendose = True
            
            elif tecla == ord('a'):
                print("Izquierda")
                carrito.avanzar(0, 0.5)
                ultim_check = time.time()
                moviendose = True
            
            elif tecla == ord('d'):
                print("Derecha")
                carrito.avanzar(0.5, 0)
                ultim_check = time.time()
                moviendose = True
            
            elif tecla == ord(' '):
                carrito.detener()
                moviendose = False # Si frenas manual, cancelamos el timer
                print("Parada de emergencia")

            elif tecla == 27: # ESC
                break

    except KeyboardInterrupt:
        if carrito is not None:
            carrito.detener()
            carrito.apagar_driver()
            print("\nPrograma terminado, carrito detenido")

    except Exception as e:
        print(f"Ocurrio un erro: {e}")
        carrito.apagar_driver()

    finally:
        #Lo detenemos totalmente
        if carrito is not None:
            carrito.detener()
            carrito.apagar_driver()
            print("\nPrograma terminado y driver en Standby")
            
