from gpiozero import Robot, Motor, OutputDevice

class Carro:
    def __init__(self, left_pins, right_pins, stby_pin, invertir_stby=False):
        self.stby = OutputDevice(stby_pin, active_high=not invertir_stby)
        self.activar_driver()

        motor_izq = Motor(forward=left_pins[0], backward=left_pins[1], enable=left_pins[2])
        motor_der = Motor(forward=right_pins[0], backward=right_pins[1], enable=right_pins[2])

        self.robot = Robot(left=motor_izq, right=motor_der)

    def _clamp(self, v, minimo=-1.0, maximo=1.0):
        return max(minimo, min(maximo, v))

    def mover(self, vel_izq, vel_der):
        """
        Control diferencial del robot.
        Entradas en rango [-1, 1]
        """
        v_i = self._clamp(vel_izq)
        v_d = self._clamp(vel_der)

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
    pines_izq = (17, 27, 12)
    pines_der = (23, 22, 13)    
    pin_stby = 24

    carrito = Carro(pines_izq, pines_der, pin_stby)