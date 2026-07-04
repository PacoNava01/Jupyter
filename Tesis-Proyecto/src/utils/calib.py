import cv2
import numpy as np
import os

def aplicar_calibracion_en_vivo(path_datos, width=640, height=480):
    """
    Carga los parámetros ópticos de la cámara y despliega un flujo de video
    aplicando la rectificación geométrica en tiempo real.
    """
    # Verificación y carga de archivos
    if not os.path.exists(path_datos):
        print(f"Error: No se encontró el archivo de calibración en: {path_datos}")
        return

    print("Cargando parámetros geométricos...")
    datos = np.load(path_datos)
    mtx = datos['matriz']
    dist = datos['distorsion']

    # Inicializar la cámara (VideoCapture estándar de OpenCV)
    # Cambiar el índice si utilizas un pipeline específico de libcamera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Validar acceso al hardware
    ret, frame_test = cap.read()
    if not ret:
        print("Error: No se puede acceder al flujo de la cámara.")
        cap.release()
        return

    #  Optimización de la matriz y precálculo de mapas (Se ejecuta UNA sola vez)
    # alpha=0 estira la imagen para eliminar por completo los bordes negros de la distorsión
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 0, (width, height))
    
    # initUndistortRectifyMap calcula la transformación en punto flotante de 32 bits para máxima velocidad
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (width, height), cv2.CV_32FC1)
    
    print("\n--- Sistema Óptico Rectificado ---")
    print("Mapas de distorsión calculados en memoria con éxito.")
    print("Presiona la tecla 'Enter' o 'q' en la ventana de video para salir.\n")

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            print("Se perdió la señal de la cámara.")
            break

        # Rotar el cuadro si la cámara está montada de forma invertida en el hardware
        frame_raw = cv2.rotate(frame_raw, cv2.ROTATE_180)

        # 4. Aplicación de la calibración mediante Remapeo (Operación ultra rápida)
        frame_corregido = cv2.remap(frame_raw, mapx, mapy, cv2.INTER_LINEAR)

        # Opcional: Recortar los bordes remanentes si la Región de Interés (ROI) es válida
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            frame_corregido = frame_corregido[y:y+h_roi, x:x+w_roi]
            # Redimensionar al tamaño original de despliegue para la comparación lado a lado
            frame_corregido = cv2.resize(frame_corregido, (width, height))

        # 5. Visualización comparativa en tiempo real
        # Concatenación horizontal: Izquierda original, Derecha corregida
        comparacion = np.hstack((frame_raw, frame_corregido))
        
        # Líneas de referencia para comprobar la rectitud de la geometría
        cv2.line(comparacion, (0, height // 2), (width * 2, height // 2), (0, 255, 255), 1)
        cv2.putText(comparacion, "ORIGINAL (Con Distorsion)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(comparacion, "CORREGIDA (Lineal)", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Prueba de Calibracion Óptica", comparacion)

        # Salir con 'q' o 'Enter' (Código ASCII 13)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 13:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Flujo terminado correctamente.")

if __name__ == "__main__":
    # Ruta absoluta hacia tu archivo de datos para evitar problemas de directorios de ejecución
    RUTA_CALIBRACION = "/home/pacon/Tesis_pacon/Jupyter/Tesis-Proyecto/data/calibracion.npz"
    
    # Ejecutar la prueba con la resolución nativa de tu bucle de control
    aplicar_calibracion_en_vivo(RUTA_CALIBRACION, width=640, height=480)