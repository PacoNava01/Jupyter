import numpy as np
import cv2
import glob

# Parametros del tablero (Esquinas internas: donde se cruzan los cuadros negros y blancos)
columnas_internas = 10
filas_internas = 8
cuadro_size_mm = 15

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
ruta_carpeta = '/home/pacon/Tesis_pacon/Jupyter/Tesis-Proyecto/data/ArUcocalib/'
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

#Guardamos la matriz 
'''
Procesando calibración con 33 imágenes válidas...

--- RESULTADOS DE LA CALIBRACIÓN ---
Matriz Intrínseca (mtx):
 [[918.7465565    0.         270.87358098]
 [  0.         920.61622722 223.78382511]
 [  0.           0.           1.        ]]

Coeficientes de Distorsión (dist):
 [[-0.50478208  0.86612519  0.00573664  0.00325532 -2.1152841 ]]

Error de reproyección total: 1.909683550439653
'''