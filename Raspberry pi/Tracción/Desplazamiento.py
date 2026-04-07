import cv2
from gpiozero import Robot, Motor, OutputDevice
from time import sleep

class Carro:
    def __init__(self, left_pins, right_pins, stby_pin):
        """
        left_pins y right_pins: (adelante, atrás, pwm)
        stby_pin: GPIO para el pin Standby del TB6612FNG
        """
        # Configuramos el pin Standby y lo ponemos en HIGH para activar el driver
        self.stby = OutputDevice(stby_pin)
        self.stby.on() 
        
        # El TB6612FNG usa la misma lógica de pines que el L298N para gpiozero
        motor_izq = Motor(forward=left_pins[0], backward=left_pins[1], enable=left_pins[2])
        motor_der = Motor(forward=right_pins[0], backward=right_pins[1], enable=right_pins[2])
        
        self.robot = Robot(left=motor_izq, right=motor_der)

    def avanzar(self, vel_izq, vel_der):
        v_izq = max(0, min(1, vel_izq))
        v_der = max(0, min(1, vel_der))
        self.robot.left_motor.forward(v_izq)
        self.robot.right_motor.forward(v_der)

    def retroceder(self, vel_izq, vel_der):
        v_izq = max(0, min(1, vel_izq))
        v_der = max(0, min(1, vel_der))
        self.robot.left_motor.backward(v_izq)
        self.robot.right_motor.backward(v_der)
    
    def girar(self,vel_izq,vel_der):
        #Obtenemos valores absolutos y los limitamos a [0, 1]
        v_izq_abs = max(0, min(1, abs(vel_izq)))
        v_der_abs = max(0, min(1, abs(vel_der)))

        # Caso giro sobre su eje (Derecha): Izquierda (+) y Derecha (-)
        if vel_izq > 0 and vel_der < 0:
            self.detener()
            self.robot.left_motor.forward(v_izq_abs)
            self.robot.right_motor.backward(v_der_abs)

        # Caso giro sobre su eje (Izquierda): Izquierda (-) y Derecha (+)
        elif vel_izq < 0 and vel_der > 0:
            self.detener()
            self.robot.left_motor.backward(v_izq_abs)
            self.robot.right_motor.forward(v_der_abs)
        
        else:
            #No hacemos nada
            pass

    def detener(self):
        self.robot.stop()

    def apagar_driver(self):
        # Opcional: pone el driver en modo bajo consumo
        self.stby.off()

#--- Playground ---

if __name__ == "__main__":
    # Pines: (IN1, IN2, PWM) para cada lado
    pines_izq = (17, 27, 12)
    pines_der = (23, 22, 13)    
    pin_stby = 24  # Conecta este pin al STBY del driver
    
    carrito = None 
    try:
        carrito = Carro(pines_izq, pines_der, pin_stby)
        print("Robot activado. Avanzando...")
        
        while True:
            carrito.avanzar(0.5, 0.5) # 80% de velocidad
            sleep(1)
          
    except KeyboardInterrupt:
        if carrito is not None:
            carrito.detener()
            carrito.apagar_driver()
            print("\nPrograma terminado y driver en Standby")

    except Exception as e:
        print(f"Ocurrió un error: {e}")