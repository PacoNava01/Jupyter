import camara
import cv2
import cv2.aruco as aruco

# Configuración del diccionario y parámetros de detección
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Crear detector
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Inicializar cámara
cam = camara.init_cam()

if cam is None:
    exit()

print('Presiona q para salir...')

while True:

    # Capturar frame
    frame = cam.capture_array()

    # Rotar imagen
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar marcadores
    corners, ids, rejected = detector.detectMarkers(gray)

    # Si se detectan marcadores
    if ids is not None:

        # Dibujar marcadores detectados
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Mostrar IDs detectados
        for i in range(len(ids)):
            print(f"ID detectado: {ids[i][0]}")

    # Mostrar frame
    cv2.imshow('ArUco Hello World', frame)

    # Salir con la tecla q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cam.stop()
cv2.destroyAllWindows()