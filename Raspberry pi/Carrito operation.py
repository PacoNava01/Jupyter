#-------------LIBRERIAS-------------

#Librerias para sistema
import os 
import time
from time import sleep
from pynput import keyboard

#Librerias para raspberry/motores-servos
from gpiozero import Motor
from adafruit_servokit import ServoKit #Util para controlar el driver PCA9685

#Librerias de vision artificial
import cv2
from picamera2 import Picamera2

#Librerias para matematicas
import numpy as np

#-------------Configuración para la camara-------------
def init_camara():
    '''Funcion para configurar e inicializar la camara'''
    try:
        picam2 = Picamera2() #Inicializamos un objeto tipo camara
        
        #Creamos la configuración 
        config = picam2.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )

        picam2.configure(config) #Aplicamos la configuración
        picam2.start() #Iniciamos la camara

        sleep(2) #Damos tiempo a la camara para que se estabilice el sensor

        #Capturamos un frame para confirmar que la camara esta funcionando
        frame = picam2.capture_array() #Capturamos un frame

        if frame is None:
            raise Exception("No se pudo capturar un frame de la camara.")

        print("Camara inicializada correctamente.")
        return picam2 #Devolvemos el objeto camara para usarlo en el programa principal

    except Exception as e:
        print(f"Error al inicializar la camara: {e}")
        return None

def capture_frame(camara):
    '''
    Funcion para capturar 
    un frame de la camara

    camara: objeto camara inicializado
    '''
    return camara.capture_array() #Capturamos un frame y lo devolvemos

def ventanas(nombre,frame):
    '''
    Funcion para crear ventanas de 
    visualizacion 
    nombre: nombre de la ventana
    frame: imagen a mostrar en la ventana
    '''
    cv2.namedWindow(nombre, cv2.WINDOW_NORMAL) #Creamos una ventana redimensionable
    cv2.imshow(nombre, frame) #Mostramos el frame en la ventana


#-------------Configuración para los motores y servos-------------
#Servomotores
def init_servos(num_servos=2):
    '''
    Funcion para configurar e 
    inicializar los servomotores
    
    num_servos: numero de servos a configurar (default 2)
    '''
    kit = ServoKit(channels=16) #Inicializamos el driver PCA9685 con 16 canales
    for i in range(num_servos): #Configuramos los primeros num_servos canales para los servos
        kit.servo[i].set_pulse_width_range(500, 2500) #Configuramos el rango de pulsos para cada servo
        kit.servo[i].actuation_range = 180 #Configuramos el rango de movimiento de cada servo a 180 grados

    for i in range(num_servos): #Inicializamos cada servo en la posición central (90 grados)
        kit.servo[i].angle = 90

    print("Servos inicializados correctamente.")
    return kit #Devolvemos el objeto kit para usarlo en el programa principal

def set_servo_angle(kit, servo_index, angle):
    '''
    Funcion para establecer el ángulo de un servo específico

    kit: objeto kit de servos inicializado
    servo_index: índice del servo al que se le quiere cambiar el ángulo
    angle: ángulo deseado para el servo (0-180 grados)
    '''
    if 0 <= angle <= 180:
        kit.servo[servo_index].angle = angle #Establecemos el ángulo del servo
        print(f"Servo {servo_index} establecido a {angle} grados.")
    else:
        print("Ángulo fuera de rango. Debe estar entre 0 y 180 grados.")

#Motores DC
def init_motor():
    '''
    Funcion para configurar 
    e inicializar los motores DC

    fordward: pin GPIO para movimiento hacia adelante
    backward: pin GPIO para movimiento hacia atrás
    '''
    #Configuramos los motores con los pines GPIO 

    motor1 = Motor(forward=17, backward=22) 
    motor2 = Motor(forward=27, backward=23) 

    #Se detienen los motores al inicio para evitar movimientos no deseados
    motor1.stop() 
    motor2.stop() 

    print("Motores DC inicializados correctamente.")
    return motor1, motor2 #Devolvemos los objetos de los motores para usarlos en el programa principal

# Variables para controlar el tiempo de los servos sin detener el video
ultimo_cambio = time.time()
estado_servo = 0


def Power_stop(kit,motores):
    '''
    Funcion para detener TODO
    '''
   

   #EJERCICIO 1

def Stress(kit, tecla,angulo_v,angulo_h, paso=5):
    '''
    Una pequeña rutina que aumente el ángulo de 
    5 en 5 grados cada vez que presiones una tecla 
    (por ejemplo, la 'a' para aumentar, 'd' para disminuir).

    kit: objeto kit de servos inicializado
    tecla: tecla presionada para controlar el servo
    paso: cantidad de grados a aumentar o disminuir con cada tecla (default 5)
    actual_v: ángulo actual del servo vertical
    actual_h: ángulo actual del servo horizontal
    '''
    nuevo_angulo_v = angulo_v
    nuevo_angulo_h = angulo_h
    
    # Comparamos con ord() porque cv2.waitKey devuelve enteros
    if tecla == ord('a'):
        nuevo_angulo_h = min(170, nuevo_angulo_h + paso)
    elif tecla == ord('d'):
        nuevo_angulo_h = max(30, nuevo_angulo_h - paso) 
    elif tecla == ord('w'):
        nuevo_angulo_v = min(170, nuevo_angulo_v + paso)
    elif tecla == ord('s'):
        nuevo_angulo_v = max(30, nuevo_angulo_v - paso)

    #Ajustamos los servos individualmente
    if nuevo_angulo_h != angulo_h:
        set_servo_angle(kit, 0, nuevo_angulo_h)    

    if nuevo_angulo_v != angulo_v:
        set_servo_angle(kit, 1, nuevo_angulo_v)    

    return nuevo_angulo_v,nuevo_angulo_h

def Control_motores(motores, tecla):

#------------- PLAYGROUND -------------
try:
    os.system('clear')
    camara = init_camara()
    kit = init_servos()

    angulo_servo_v = 90 
    angulo_servo_h = 90 
    set_servo_angle(kit, 0, angulo_servo_v) 
    set_servo_angle(kit, 1, angulo_servo_h)
    
    while True:
        frame = capture_frame(camara)
        ventanas("Camara", frame)

        # 1. CAPTURAMOS LA TECLA UNA SOLA VEZ
        key = cv2.waitKey(1) & 0xFF 

        # 2. ACTUALIZAMOS EL ÁNGULO USANDO EL RETORNO DE LA FUNCIÓN
        angulo_servo_v, angulo_servo_h = Stress(kit, key, angulo_servo_v, angulo_servo_h) 
        
        # 3. SALIDA
        if key == ord('q'):
            break
    
finally:
    cv2.destroyAllWindows()
    if camara is not None:
        camara.stop()
    # Freno de seguridad para los motores DC si los añades luego
    set_servo_angle(kit, 0, 90)
    set_servo_angle(kit, 1, 90)
    print("Programa terminado")
