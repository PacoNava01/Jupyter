import cv2
from picamera2 import Picamera2
import time
import os
import glob 
import numpy as np


# Parametros
intervalo = 3
nombre_ventana = "Frame"
# Directorio destino
DICT_DESTINO = r"/home/pacon/Tesis_pacon/Jupyter/Tesis-Proyecto/data/ArUcocalib"

def init_cam():
    try:
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        print("Cámara inicializada correctamente.")
        return cam
    except Exception as e:
        print(f"Error al inicializar la cámara: {e}")
        return None

def captura_photos(Directorio,Intervalo,N_Ventana):
    '''Capturamos una foto cada x segundos mientras se "graba un video"'''
    contador = 1

    ultimo_chek = time.time()

    camara = init_cam()
    if camara is None: exit()

    cv2.namedWindow(N_Ventana)
    while True:
        frame = camara.capture_array() 
        frame = cv2.rotate(frame, cv2.ROTATE_180) # frame para mostrar
        
        # Clonamos el frame original ANTES de pintarle texto para guardarlo limpio
        frame2save = frame.copy()

        ahora = time.time()

        if ahora - ultimo_chek >= Intervalo:
            # Nombre del archivo con la ruta completa
            nombre_archivo = f"frame_calib_{contador}.jpg"
            ruta_completa = os.path.join(Directorio, nombre_archivo)

            #Guardamos la imagen
            cv2.imwrite(ruta_completa, frame2save)
            print(f"Foto guardada: {ruta_completa}")

            ultimo_chek = ahora
            contador += 1

        # Mostramos en pantalla el número de la PRÓXIMA foto que se va a tomar
        cv2.putText(frame, f"Prox. Foto: #{contador}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow(N_Ventana, frame)

        # Detecta la tecla Enter (13) para salir
        if cv2.waitKey(1) & 0xFF == 13: break
    camara.stop()
    cv2.destroyAllWindows()



# Parametros del tablero (Esquinas internas: donde se cruzan los cuadros negros y blancos)
columnas_internas = 10
filas_internas = 8
cuadro_size_mm = 15
# Cargar las imagenes de calibracion con la ruta correcta
ruta_carpeta = r'/home/pacon/Tesis_pacon/Jupyter/Tesis-Proyecto/data/ArUcocalib/'
ruta_save = r'Tesis-Proyecto\data'

def calibracion(ruta_carpeta,ruta_save,columnas_internas = 10,filas_internas = 8,cuadro_size_mm = 15):
    # Criterios de terminacion para la optimizacion (CORREGIDO)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Preparar los puntos de objeto (0,0,0), (15,0,0), (30,0,0),...
    objp = np.zeros((filas_internas * columnas_internas, 3), np.float32)
    # Generamos la malla adaptada correctamente a las dimensiones
    objp[:, :2] = np.mgrid[0:columnas_internas, 0:filas_internas].T.reshape(-1, 2)
    objp *= cuadro_size_mm

    objpoints = []  # Puntos 3D en el mundo real
    imgpoints = []  # Puntos 2D en el plano de la imagen

    # Cargar las imagenes de calibracion con la ruta correcta
    imagenes = glob.glob(ruta_carpeta + 'frame_calib_*.jpg')

    if len(imagenes) == 0:
        print("Error: No se encontraron imágenes con el patrón 'frame_calib_*.jpg'")
        exit()

    shape_imagen = None

    for frame in imagenes:
        img = cv2.imread(frame)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape_imagen = gray.shape[::-1]  # Guardamos el tamaño (ancho, alto)

        # Buscar las esquinas del tablero
        ret, corners = cv2.findChessboardCorners(gray, (columnas_internas, filas_internas), None)

        if ret == True:
            objpoints.append(objp)
            # Refinar las coordenadas de las esquinas para mayor precision
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Opcional: Dibujar y mostrar las esquinas
            cv2.drawChessboardCorners(img, (columnas_internas, filas_internas), corners2, ret)
            cv2.imshow('Calibrando...', img)
            cv2.waitKey(500)

    # Al terminar el bucle, cerramos las ventanas de muestra
    cv2.destroyAllWindows()

    # --- LA CALIBRACIÓN VA AQUÍ (FUERA DEL BUCLE) ---
    if len(objpoints) > 0:
        print(f"Procesando calibración con {len(objpoints)} imágenes válidas...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape_imagen, None, None)
        print("\n--- RESULTADOS DE LA CALIBRACIÓN ---")
        print("Matriz Intrínseca (mtx):\n", mtx)
        print("\nCoeficientes de Distorsión (dist):\n", dist)
        print(f"\nError de reproyección total: {ret}")
    else:
        print("Error: No se pudieron detectar las esquinas en ninguna de las imágenes.")

    ruta_2save = os.path.join(ruta_save,"Parametros_calibracion")
    #Guardamos la matriz fr la camara y los coeficientes de distorsion 
    np.savez(ruta_2save, matriz=mtx,distorsion=dist)
    print(f"Parametros de caslibracion guardados exitosamente en: {ruta_2save}")    

        

if __name__ == '__main__':
    print("\n=== MENÚ PRINCIPAL ===")
    print("1. Tomar nuevas fotos para calibración")
    print("2. Usar imágenes existentes para calibración")
    print("3. Salir")

    try:
        opcion = int(input("Selecciona una opción (1-3): "))
    except ValueError:
        print("Entrada inválida. Debes ingresar un número.")
        exit()

    if opcion == 1:
        print("Iniciando captura de fotos...")
        captura_photos(DICT_DESTINO, intervalo, nombre_ventana)

    elif opcion == 2:
        print("Iniciando calibración con imágenes existentes...")
        calibracion(ruta_carpeta, ruta_save)

    elif opcion == 3:
        print("Saliendo del programa.")
        exit()

    else:
        print("Opción no reconocida. Intenta de nuevo.")

        