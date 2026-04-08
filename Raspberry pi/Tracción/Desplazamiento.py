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
        
    #Metodo principal
    def mover(self, vel_izq, vel_der):
        '''
        control directo tipo joystick
        -1.0 a 1.0
        Negativo = atrás
        Positivo = adelante
        '''
        
        vel_izq = max(-1, min(1, vel_izq))
        vel_der = max(-1, min(1, vel_der))

        self.robot.left_motor.value = vel_izq
        self.robot.right_motor.value = vel_der
        
        
    # --- Metodos simples ---
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

    def apagar_driver(self):
        # Opcional: pone el driver en modo bajo consumo
        self.detener()
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