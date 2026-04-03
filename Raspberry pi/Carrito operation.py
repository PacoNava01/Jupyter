#-------------LIBRERIAS-------------

# Librerías para sistema
import os 
import time
from time import sleep

# Librerías para raspberry/motores-servos
from gpiozero import Motor
from adafruit_servokit import ServoKit

# Librerías de visión artificial
import cv2
from picamera2 import Picamera2

# Librerías para matemáticas
import numpy as np

#-------------CONFIGURACIÓN CÁMARA-------------
def init_camara():
    """
    Inicializa y configura la cámara.
    Devuelve el objeto cámara si todo funciona,
    o None si ocurre algún error.
    """
    try:
        picam2 = Picamera2()

        config = picam2.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )

        picam2.configure(config)
        picam2.start()

        sleep(1)  # Espera para estabilización del sensor

        frame = picam2.capture_array()
        if frame is None:
            raise Exception("No se pudo capturar frame inicial.")

        print("Cámara inicializada correctamente.")
        return picam2

    except Exception as e:
        print(f"Error al inicializar cámara: {e}")
        return None


def capture_frame(camara):
    """
    Captura un frame de la cámara.
    """
    return camara.capture_array()


def mostrar_frame(nombre, frame):
    """
    Solo muestra el frame.
    La ventana debe crearse UNA sola vez fuera del loop.
    """
    cv2.imshow(nombre, frame)


def info_frame_angle(frame, ang_h, ang_v):
    """
    Dibuja información sobre el frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 5, 255)

    cv2.putText(frame, f"Servo Horizontal: {ang_h} deg",
                (10, 30), font, 0.7, color, 2)

    cv2.putText(frame, f"Servo Vertical: {ang_v} deg",
                (10, 60), font, 0.7, color, 2)

    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    length = 20
    cv2.line(frame, (center_x, center_y - length),
             (center_x, center_y + length), (0, 255, 0), 2)

    cv2.line(frame, (center_x - length, center_y),
             (center_x + length, center_y), (0, 255, 0), 2)


#-------------CONFIGURACIÓN SERVOS-------------
def init_servos(num_servos=2):
    """
    Inicializa PCA9685 y configura los servos.
    """
    try:
        kit = ServoKit(channels=16)

        for i in range(num_servos):
            kit.servo[i].set_pulse_width_range(500, 2500)
            kit.servo[i].actuation_range = 180
            kit.servo[i].angle = 90  # Posición inicial segura

        print("Servos inicializados correctamente.")
        return kit

    except Exception as e:
        print(f"Error al inicializar servos: {e}")
        return None


def set_servo_angle(kit, servo_index, angle):
    """
    Establece el ángulo de un servo.
    Incluye validación de rango para evitar daños físicos.
    """
    if kit is None:
        return

    if 0 <= angle <= 180:
        kit.servo[servo_index].angle = angle
    else:
        print("Ángulo fuera de rango (0-180).")


#-------------CONFIGURACIÓN MOTORES-------------
M1_FWD, M1_BWD = 17, 27
M2_FWD, M2_BWD = 22, 23


def init_motores():
    """
    Inicializa motores DC.
    Devuelve lista [motor1, motor2] o None si falla.
    """
    try:
        motor1 = Motor(forward=M1_FWD, backward=M1_BWD)
        motor2 = Motor(forward=M2_FWD, backward=M2_BWD)

        motor1.stop()
        motor2.stop()

        print("Motores inicializados correctamente.")
        return [motor1, motor2]

    except Exception as e:
        print(f"Error al inicializar motores: {e}")
        return None


def test_motores(motores, tecla):
    """
    Control básico con teclado.
    """
    if motores is None:
        return

    motor1, motor2 = motores

    if tecla == ord('i'):
        motor1.forward()
        motor2.forward()
    elif tecla == ord('k'):
        motor1.backward()
        motor2.backward()
    elif tecla == ord('j'):
        motor1.backward()
        motor2.forward()
    elif tecla == ord('l'):
        motor1.forward()
        motor2.backward()
    elif tecla == ord('x'):
        motor1.stop()
        motor2.stop()


#-------------CONTROL SERVOS-------------
def controlar_servos(kit, tecla, ang_v, ang_h, paso=5):
    """
    Modifica ángulos según tecla.
    Mantiene límites seguros para evitar forzar el servo.
    """

    nuevo_v = ang_v
    nuevo_h = ang_h

    if tecla == ord('a'):
        nuevo_h = min(170, ang_h + paso)
    elif tecla == ord('d'):
        nuevo_h = max(30, ang_h - paso)
    elif tecla == ord('w'):
        nuevo_v = min(170, ang_v + paso)
    elif tecla == ord('s'):
        nuevo_v = max(30, ang_v - paso)

    # Solo mover si realmente cambió el valor
    if nuevo_h != ang_h:
        set_servo_angle(kit, 0, nuevo_h)

    if nuevo_v != ang_v:
        set_servo_angle(kit, 1, nuevo_v)

    return nuevo_v, nuevo_h

#Imporamos de otro programa
from Detector2 import obtener_mask, procesar_contornos, low_red1, up_red1, low_red2, up_red2
# --- Rangos Rojos (Ajustados para RGB -> HSV) ---
low_red1 = np.array([0, 100, 50], dtype=np.uint8)
up_red1 = np.array([10, 255, 255], dtype=np.uint8)
low_red2 = np.array([170, 100, 50], dtype=np.uint8)
up_red2 = np.array([180, 255, 255], dtype=np.uint8)




#-------------PROGRAMA PRINCIPAL-------------
camara = None
kit = None
motores = None

try:
    os.system('clear')

    camara = init_camara()
    
    if camara is None:
        raise Exception("No se pudo iniciar la cámara.")

    kit = init_servos()
    if kit is None:
        raise Exception("No se pudo iniciar servos.")

    #motores = init_motores()

    angulo_v = 90
    angulo_h = 90

    # Crear ventana UNA sola vez
    cv2.namedWindow("Deteccion",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mascara",cv2.WINDOW_NORMAL)


    while True:

        frame = capture_frame(camara)
        if frame is None:
            print("Frame inválido.")
            break
        
        mask1 = obtener_mask(frame, low_red1, up_red1)
        mask2 = obtener_mask(frame, low_red2, up_red2)
        mask_red = cv2.add(mask1, mask2)

        frame_con_deteccion, centroide = procesar_contornos(mask_red, frame.copy(), min_area=500)

        if centroide:
            cv2.putText(frame_con_deteccion, f"Pos: {centroide}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Dibujamos información sobre el frame
        info_frame_angle(frame_con_deteccion, angulo_h, angulo_v)
        
        mostrar_frame("Deteccion", frame_con_deteccion)
        mostrar_frame("Mascara", mask_red)

      
        mostrar_frame("Camara", frame)

        # IMPORTANTE:
        # waitKey debe ir después de imshow
        key = cv2.waitKey(1) & 0xFF

        # Control motores y servos
        #test_motores(motores, key)
        angulo_v, angulo_h = controlar_servos(
            kit, key, angulo_v, angulo_h
        )

        if key == 13:
            break

finally:
    print("Cerrando sistema...")

    cv2.destroyAllWindows()

    if camara is not None:
        camara.stop()

    if kit is not None:
        set_servo_angle(kit, 0, 90)
        set_servo_angle(kit, 1, 90)

    if motores is not None:
        for motor in motores:
            motor.stop()

    print("Programa terminado correctamente.")