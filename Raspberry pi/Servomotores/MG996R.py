from adafruit_servokit import ServoKit
import time

# Inicializamos el driver (especificamos que tiene 16 canales)
kit = ServoKit(channels=16)

#Configuracion de servos
for i in range(0,2):
    kit.servo[i].set_pulse_width_range(600, 2400)
    kit.servo[i].actuation_range = 160

#Renombramos servos de acuerdo a su función
longitud = kit.servo[0]
latitud = kit.servo[1]

# Mover el servo en el canal 0 a 90 grados
print("Moviendo a 90 grados...")
longitud.angle = 90
latitud.angle = 90

time.sleep(2)

# Mover a 0 grados
print("Moviendo a 0 grados...")
longitud.angle = 0
latitud.angle = 45

print("Giro realizado")