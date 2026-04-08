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
        intervalo = 5
        ultim_check = time.time()
        
        
        impulso_activo = False
        impulso_inicio = 0
        duracion_impulso = 0.3 #segundos
        
        vel_giro_normal = 0.6
        lado_izquierdo = 0
        lado_derecho = 0

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
                carrito.avanzar(0.5) #carrito.mover(0.9,0.9) tambien es posible
                ultim_check = time.time() 
                moviendose = True # <--- Activamos la cuenta regresiva
                
                #Control no bloqueante
            if impulso_activo:
                # Si el tiempo del impulso terminó, bajar a velocidad normal
                if tiempo_actual - impulso_inicio >= duracion_impulso:
                  carrito.mover(lado_izquierdo * vel_giro_normal, lado_derecho * vel_giro_normal)
                  impulso_activo = False  # desactivamos impulso
                  
               
            elif tecla == ord('s'):
                print("Retrocediendo...")
                carrito.retroceder(0.5) #carrito.mover(-0.9,-0.9) tambien es posible
                ultim_check = time.time()
                moviendose = True
            
            elif tecla == ord('a'):
                print("Izquierda")
                carrito.mover(-1, 1)
                ultim_check = time.time()
                moviendose = True
                impulso_inicio = tiempo_actual
            
            elif tecla == ord('d'):
                print("Derecha")
                carrito.mover(1, -1)
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
            
