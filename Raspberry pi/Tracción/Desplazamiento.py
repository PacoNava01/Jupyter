class Carro:
    # Nuevos coeficientes ajustados para escala 0.0 - 1.0
    COEFS = {
        'a': 0.9356779218031275,
        'b': -1.864449111530371,
        'c': 1.1723900277729624,
        'd': -0.2070794160209165
    }

    def __init__(self, left_pins, right_pins, stby_pin):
        self.stby = OutputDevice(stby_pin)
        self.stby.on() 
        
        motor_izq = Motor(forward=left_pins[0], backward=left_pins[1], enable=left_pins[2])
        motor_der = Motor(forward=right_pins[0], backward=right_pins[1], enable=right_pins[2])

        self.robot = Robot(left=motor_izq, right=motor_der)

    # ------------------------
    # Utilidades internas
    # ------------------------
    def _clamp(self, v, minimo=-1.0, maximo=1.0):
        return max(minimo, min(maximo, v))

    def C_compensacion(self, pwm_objetivo):
        ''' 
        Calcula el factor de corrección usando la escala 0-1.
        Si pwm_objetivo = 1.0 (100%), el error será ~0.036
        '''
        # Extraemos los coeficientes
        a, b, c, d = self.COEFS.values()
        
        # Limitamos el valor al rango de tus datos experimentales (0.4 a 1.0)
        # Esto evita que el polinomio "explote" fuera de ese rango
        x = max(0.4, min(1.0, pwm_objetivo))
        
        # Evaluamos el error esperado
        error = a*x**3 + b*x**2 + c*x + d
        
        # El factor de compensación es (1 - error)
        return 1 - error
    
    def compensar_derecho(self,v):
        signo = 1 if v >= 0 else -1
        magnitud = abs(v)
        
        # No compensar en zona baja (ruido/fricción domina)
        if magnitud < 0.4:
            return v
        
        factor = self.C_compensacion(magnitud)
        return signo * magnitud
    
    # ------------------------
    # Metodos principales
    # ------------------------

    def mover(self, vel_izq, vel_der):
        """
        Método principal de movimiento.
        Recibe valores en rango [-1, 1].
        Aplica:
        - clamp
        - compensación derecha
        - clamp final
        """

        v_i = self._clamp(vel_izq)
        v_d = self._clamp(vel_der)

        # compensación solo lado derecho
        v_d = self._compensar_derecho(v_d)

        # clamp final por seguridad
        v_d = self._clamp(v_d)

        self.robot.left_motor.value = v_i
        self.robot.right_motor.value = v_d

    # ------------------------
    # Métodos auxiliares
    # ------------------------
    def detener(self):
        self.robot.stop()

    def apagar_driver(self):
        self.detener()
        self.stby.off()

    
    def mover(self, vel_izq, vel_der):
        # Validamos que los valores no excedan el rango de gpiozero
        v_i = max(-1, min(1, vel_izq))
        v_d = max(-1, min(1, vel_der))
        factor = self.C_compensacion(v_d)
        v_d = v_d * factor
        
        
        self.robot.left_motor.value = v_i
        self.robot.right_motor.value = v_d
    
    def detener(self):
        self.robot.stop()

    def apagar_driver(self):
        # Opcional: pone el driver en modo bajo consumo
        self.detener()
        self.stby.off()

if __name__ == "main":
    # Pines: (IN1, IN2, PWM) para cada lado
    pines_izq = (17, 27, 12)
    pines_der = (23, 22, 13)    
    pin_stby = 24  # Conecta este pin al STBY del driver

    carrito = None
    try:
        carrito = Carro(pines_izq,pines_der)
        print("Robot activado...")

    except KeyboardInterrupt:
        if carrito is not None:
            carrito.detener()
            carrito.apagar_driver()
            print("\nPrograma terminado y driver en Standby")
    
    except Exception as e:
        print(f"Ocurrió un error: {e}")