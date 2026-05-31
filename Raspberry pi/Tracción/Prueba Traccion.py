#from Hardware.Motores_DC.Desplazamiento import carro
import numpy as np
import cv2
import time

if __name__ == "__main__":
    pines_izq = (17,27,12)
    pines_der = (23,22,13)
    pin_stby = 24

    carrito = None
    
    cv2.namedWindow("Control Carrito")

    try:
        #carrito = Carro(pines_izq, pines_der, stby_pin=pin_stby)
        print("Robot activado...")

        moviendose = False
        impulso_activo = False
        intervalo = 2
        ultim_check = time.time()

        while True:
            frame_dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow("Control Carrito", frame_dummy)

            tecla = cv2.waitKey(1) & 0xFF
            tiempo_actual = time.time()

            # --- TEMPORIZADOR ---
            if moviendose and (tiempo_actual - ultim_check) > intervalo:
                carrito.detener()
                print("Intervalo cumplido: Detenido")
                moviendose = False

            # --- CONTROLES ---
            if tecla == ord('w'):
                carrito.mover(0.5,0.5)
                print("Avanzando...")
                ultim_check = tiempo_actual
                moviendose = True

            elif tecla == ord('s'):
                carrito.mover(-0.5,0.5)  
                print("Retrocediendo...")
                ultim_check = tiempo_actual
                moviendose = True
            
            elif tecla == ord('d'):
                carrito.mover(0.5,-0.5)  
                print("Retrocediendo...")
                ultim_check = tiempo_actual
                moviendose = True

            elif tecla == ord(' '):
                carrito.detener()
                moviendose = False
                print("Parada de emergencia")

            elif tecla == 27:
                break

    except Exception as e:
        print(f"Ocurrió un error: {e}")

    finally:
        if carrito is not None:
            carrito.detener()
            carrito.apagar_driver()
        cv2.destroyAllWindows()
        print("Sistema apagado correctamente")