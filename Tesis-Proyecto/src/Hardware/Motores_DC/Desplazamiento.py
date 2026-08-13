from gpiozero import Robot, Motor, OutputDevice
import os
import cv2
import time

class Carro:
    # Nuevos coeficientes ajustados para escala 0.0 - 1.0
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

    # ------------------------
    # Utilidades internas
    # ------------------------
    def _clamp(self, v, minimo=-1.0, maximo=1.0):
        return max(minimo, min(maximo, v))

    def _compensacion(self, pwm):
        """
        Calcula factor de compensación para magnitudes (0 a 1)
        """
        a = self.COEFS['a']
        b = self.COEFS['b']
        c = self.COEFS['c']
        d = self.COEFS['d']

        x = max(0.4, min(1.0, pwm))
        error = a*x**3 + b*x**2 + c*x + d
        return 1 - error

    def _compensar_derecho(self, v):
        signo = 1 if v >= 0 else -1
        magnitud = abs(v)

        if magnitud < 0.4:
            return v

        factor = self._compensacion(magnitud)
        return signo * (magnitud * factor - 0.2)
        

    # ------------------------
    # Métodos principales
    # ------------------------
    def mover(self, vel_izq, vel_der):
        """
        Control diferencial del robot.
        Entradas en rango [-1, 1]
        """

        v_i = self._clamp(vel_izq)
        v_d = self._clamp(vel_der)

        # Aplicar compensación solo al motor derecho
        v_d = self._compensar_derecho(v_d)

        # Clamp final
        v_d = self._clamp(v_d)

        self.robot.left_motor.value = v_i
        self.robot.right_motor.value = v_d

    def avanzar(self, velocidad=0.5):
        self.mover(velocidad, velocidad)

    def retroceder(self, velocidad=0.5):
        self.mover(-velocidad, -velocidad)

    def girar_izquierda(self, velocidad=0.5):
        self.mover(-velocidad, velocidad)

    def girar_derecha(self, velocidad=0.5):
        self.mover(velocidad, -velocidad)

    def detener(self):
        self.robot.stop()

    # ------------------------
    # Control del driver
    # ------------------------
    def activar_driver(self):
        self.stby.on()

    def apagar_driver(self):
        self.detener()
        self.stby.off()

if __name__ == "__main__":
    # Pines: (IN1, IN2, PWM) para cada lado
    pines_izq = (17, 27, 12)
    pines_der = (23, 22, 13)    
    pin_stby = 24  # Conecta este pin al STBY del driver

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