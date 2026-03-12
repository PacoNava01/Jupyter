'''Codigo para poder realizar el desplazamiento 
de un robot 4WD usando la libreria gpiozero 
y un modulo de control de motores L298N'''

#--- Librerias ---
import cv2
from gpiozero import Robot,Motor
from time import sleep
import cv2


class Carro:
    def __init__(self, left_pins, right_pins):
        """
        left_pins y right_pins deben ser tuplas de 3: 
        (adelante, atrás, enable/pwm)
        Ejemplo: left_pins=(17, 27, 12)
        """
        # En lugar de pasar las tuplas directo a Robot, 
        # creamos los motores primero con sus nombres de parámetros.
        motor_izq = Motor(forward=left_pins[0], backward=left_pins[1], enable=left_pins[2])
        motor_der = Motor(forward=right_pins[0], backward=right_pins[1], enable=right_pins[2])
        
        # Ahora se los pasamos al Robot
        self.robot = Robot(left=motor_izq, right=motor_der)
        self.velocidad_izq = 0 
        self.velocidad_der = 0

    def avanzar(self, vel_izq, vel_der):
        # Ajuste y validación
        self.velocidad_izq, self.velocidad_der = max(0, min(1, vel_izq)), max(0, min(1, vel_der))
        
        self.robot.left_motor.forward(self.velocidad_izq)
        self.robot.right_motor.forward(self.velocidad_der)

    def retroceder(self, vel_izq, vel_der):
        #Ajuste y validación
        self.velocidad_izq, self.velocidad_der = max(0, min(1, vel_izq)), max(0, min(1, vel_der))
        
        self.robot.left_motor.backward(self.velocidad_izq)
        self.robot.right_motor.backward(self.velocidad_der)

    def detener(self):
        self.robot.stop()  

#--- Playground ---

'''
Condiciones:

Fuente de alimentación a 5V para el L298N, con una corriente máxima de 2A por canal.

'''

if __name__ == "__main__":
    pines_izq = (17, 27, 12)
    pines_der = (23, 22, 13)    
    
    carrito = None # Inicializamos la variable como vacía
    tecla = None
    try:
        carrito = Carro(pines_izq, pines_der)
        while True:
            carrito.avanzar(1, 1)
            sleep(10)
          
        
    except Exception as e:
        print(f"Ocurrió un error: {e}")
        
    finally:
        # Solo intentamos detener si el carrito realmente se creó
        if carrito is not None:
            carrito.detener()
            print("Programa terminado, carrito detenido")