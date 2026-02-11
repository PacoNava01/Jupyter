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

def simple():
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
        elif user_input == 'm':
            motor1.stop(); motor2.stop()
            break # Sale del bucle para volver al menú

def giro_simple():
    #Programa para girar en direcciones inicadas
    os.system('clear')
    while True:
        #Convertimos a minusculas para manejar cualquier caso
        user_input = input("Indica la dirección a la que deseas girar\nIzquierda-i\nDerecha-d ")
        print(f"Introdujiste {user_input}")
        if user_input == 'i':
            print("Girando a la derecha")
            motor1.forward()
            motor2.stop()
        elif user_input == 'd':
            print("Girando  la derecha")
            motor1.stop()
            motor2.forward()
        elif user_input == 'm':
            motor1.stop(); motor2.stop()
            break # Sale del bucle para volver al menú
        else:
            print("Comando no reconocido, deteniendo motores")
 
 # --- MENÚ PRINCIPAL ---
try:
    while True:
        os.system('clear')
        print("=== PANEL DE CONTROL DE MOTORES ===")
        print("1. Control Simple (Adelante/Atrás)")
        print("2. Control de Giros (Izquierda/Derecha)")
        print("3. Salir")
        
        opcion = input("\nSelecciona una opción: ")

        if opcion == '1':
            simple()
        elif opcion == '2':
            giro_simple()
        elif opcion == '3':
            break
        else:
            print("Opción inválida")
            sleep(1)

except KeyboardInterrupt:
    print("\n\nSaliendo de forma segura...")
finally:
    motor1.stop()
    motor2.stop()
    print("Motores apagados. ¡Adiós!")