from gpiozero import Motor,PWMOutputDevice
from time import sleep
import os 

os.system('clear')#Limpiamos lla terminal
#Configuramos los motores
motor1 = Motor(forward=17,backward=22)
motor2 = Motor(forward=27,backward=23)

#Los inicializamos en cero
motor1.stop()
motor2.stop()
try:
    while True:
        # Convertimos a minúsculas para manejar 'B' y 'b' de un solo golpe
        user_input = input("introduce algun comando \ncomandos disponibles:" \
        "\nFordward-f, Backward-b ,Stop-s\n(Ctrl+C para salir): ").lower()
        print(f"Introdujuste: {user_input}")

        if user_input == 'b':
            print("Motores en reversa")
            motor1.backward()
            motor2.backward()
        elif user_input == 'f':
            print("Motores hacia delante")
            motor1.forward()
            motor2.forward()
        elif user_input == 's':
            print("Motores detenido")
            motor1.stop()
            motor2.stop()

            
except KeyboardInterrupt:
    print("\nPrograma terminado por el usuario.")
    motor1.stop() # Buena práctica detenerlo al salir
    motor2.stop()