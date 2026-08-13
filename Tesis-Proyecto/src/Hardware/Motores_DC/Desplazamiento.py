from gpiozero import Robot, Motor, OutputDevice
<<<<<<< HEAD
import os
import cv2
import time
=======
import math
>>>>>>> 1da99ca60e510b501ffb13674dc76cc9a41a06ad

class Carro:
    COEFS = {
        'a': 0.9356779218031275,
        'b': -1.864449111530371,
        'c': 1.1723900277729624,
        'd': -0.2070794160209165
    }

    def __init__(self, left_pins, right_pins, stby_pin, invertir_stby=False):
        self.stby = OutputDevice(stby_pin, active_high=not invertir_stby)
        self.activar_driver()

        motor_izq = Motor(forward=left_pins[0], backward=left_pins[1], enable=left_pins[2])
        motor_der = Motor(forward=right_pins[0], backward=right_pins[1], enable=right_pins[2])

        self.robot = Robot(left=motor_izq, right=motor_der)

    def _clamp(self, v, minimo=-1.0, maximo=1.0):
        return max(minimo, min(maximo, v))

    def _compensacion(self, pwm):
        a, b, c, d = (self.COEFS[k] for k in ('a','b','c','d'))
        x = max(0.4, min(1.0, pwm))
        error = a*x**3 + b*x**2 + c*x + d
        return 1 - error

    def _compensar_derecho(self, v):
        magnitud = abs(v)
        if magnitud < 0.4:
            return v
        factor = self._compensacion(magnitud)
        return math.copysign(magnitud * factor - 0.2, v)

    def mover(self, vel_izq, vel_der):
        v_i = self._clamp(vel_izq)
        v_d = self._clamp(self._compensar_derecho(vel_der))
        self.robot.left_motor.value = v_i
        self.robot.right_motor.value = v_d

    def accion(self, tipo, velocidad=0.5):
        acciones = {
            'avanzar': (velocidad, velocidad),
            'retroceder': (-velocidad, -velocidad),
            'izquierda': (-velocidad, velocidad),
            'derecha': (velocidad, -velocidad)
        }
        v_i, v_d = acciones[tipo]
        self.mover(v_i, v_d)

    def detener(self):
        self.robot.stop()

    def activar_driver(self):
        self.stby.on()

    def apagar_driver(self):
        self.detener()
        self.stby.off()

if __name__ == "__main__":
<<<<<<< HEAD
    # Pines: (IN1, IN2, PWM) para cada lado
=======
>>>>>>> 1da99ca60e510b501ffb13674dc76cc9a41a06ad
    pines_izq = (17, 27, 12)
    pines_der = (23, 22, 13)    
    pin_stby = 24

<<<<<<< HEAD
    carrito = None
    try:
        print("Inicializando carrito...")
        carrito = Carro(pines_izq, pines_der, pin_stby)
        
        print("-> Avanzando...")
        #carrito.avanzar(0.5)
        #time.sleep(2)
        
        print("-> Retrocediendo...")
        #carrito.retroceder(0.5)
        #time.sleep(2)
        
        print("-> Girando a la izquierda...")
        carrito.girar_izquierda(0.4)
        time.sleep(1.5)
        
        print("-> Girando a la derecha...")
        carrito.girar_derecha(0.6)
        time.sleep(1.5)
        
        print("-> Deteniendo motores...")
        carrito.detener()

    except KeyboardInterrupt:
        print("\nPrueba interrumpida por el usuario.")
    
    finally:
        if carrito:
            print("Apagando driver y liberando pines...")
            carrito.apagar_driver()
=======
    carrito = Carro(pines_izq, pines_der, pin_stby)
>>>>>>> 1da99ca60e510b501ffb13674dc76cc9a41a06ad
