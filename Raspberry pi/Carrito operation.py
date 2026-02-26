#-------------LIBRERIAS-------------

#Librerias para sistema
import os 
from time import sleep

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




#------------- PLAYGROUND -------------
try:
    #Limpio la terminal
    os.system('clear')
    camara = init_camara() #Inicializamos la camara al inicio del programa

    while True:
        frame = capture_frame(camara) #Capturamos un frame de la camara
        ventanas("Camara",frame) #Mostramos el frame en una ventana

        if cv2.waitKey(1) & 0xFF == ord('q'): #Si presionamos 'q' salimos del loop
            break
finally:
    cv2.destroyAllWindows() #Cerramos todas las ventanas al finalizar el programa
    camara.stop() #Detenemos la camara al finalizar el programa
    print("Programa terminado")

