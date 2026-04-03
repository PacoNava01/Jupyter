import time 

intervalo = 2 # Cambié a 5 para que coincida con tu condición
ultimo_check = time.time() 


if __name__ == "__main__":
    try:
        while True:
            tiempo_actual = time.time()
            

            if (time.time() - ultimo_check) > intervalo:
                print(tiempo_actual-time.time())
                print("Intervalo alcanzado")
                ultimo_check = time.time()
    except KeyboardInterrupt:
        print("\nProgrma terminado")
        



