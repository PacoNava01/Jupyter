# --- Librerias propias---
from Desplazamiento import Carro

# --- Librerias ---
import numpy as np
import cv2
import time
import os

# --- Playground

if __name__ == "__main__":
    #(forward,backward,enable)
    pines_izq = (17,27,12)
    pines_der = (23,22,13)
    stby = 24

    carrito = None

    cv2.namedWindow("Control Carrito")

    try:
        carrito = Carro(pines_izq, pines_der, stby)
        print("Carrito instanciado. \nPresiona W, A, S, D. 'ESC' para salir.")
        
        moviendose = False
        intervalo = 5
        ultim_check = time.time()
        
        # --- CONTROL DE IMPULSO para pruebas de giro---
        impulso_activo = False
        impulso_inicio = 0
        duracion_impulso = 0.3
        
        vel_giro_normal = 0.6
        lado_izquierdo = 0
        lado_derecho = 0

        while True:
            tecla = cv2.waitKey(1) & 0xFF 
            tiempo_actual = time.time()

            # --- TEMPORIZADOR DE MEDICIÓN ---
            if moviendose and (tiempo_actual - ultim_check) > intervalo:
                carrito.detener()
                print("Intervalo cumplido: Deteniendo para medición...")
                moviendose = False
                impulso_activo = False

            # --- CONTROL DE IMPULSO (NO BLOQUEANTE) ---
            if impulso_activo:
                if tiempo_actual - impulso_inicio >= duracion_impulso:
                    carrito.mover(
                        lado_izquierdo * vel_giro_normal,
                        lado_derecho * vel_giro_normal
                    )
                    impulso_activo = False

            # --- CONTROL DE TECLADO ---
            if tecla == ord('w'):
                print("Avanzando...")
                carrito.avanzar(0.5)
                ultim_check = tiempo_actual
                moviendose = True
                impulso_activo = False

            elif tecla == ord('s'):
                print("Retrocediendo...")
                carrito.retroceder(0.5)
                ultim_check = tiempo_actual
                moviendose = True
                impulso_activo = False
            
            elif tecla == ord('a'):
                print("Izquierda (con impulso)")
                
                carrito.mover(-1, 1)  # impulso fuerte
                
                lado_izquierdo = -1
                lado_derecho = 1
                
                impulso_inicio = tiempo_actual
                impulso_activo = True
                
                ultim_check = tiempo_actual
                moviendose = True
            
            elif tecla == ord('d'):
                print("Derecha (con impulso)")
                
                carrito.mover(1, -1)
                
                lado_izquierdo = 1
                lado_derecho = -1
                
                impulso_inicio = tiempo_actual
                impulso_activo = True
                
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
            print("\nPrograma terminado, carrito detenido")

    except Exception as e:
        print(f"Ocurrio un error: {e}")
        if carrito is not None:
            carrito.apagar_driver()

    finally:
        if carrito is not None:
            carrito.detener()
            carrito.apagar_driver()
            print("\nPrograma terminado y driver en Standby")