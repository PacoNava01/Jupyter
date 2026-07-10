# =====================================================================
# MODIFICACIÓN INTELIGENTE: IMPORTACIONES PARA U-NET Y ARUCO
# =====================================================================
from Vision.camara import init_cam
from Hardware.Servomotores.MG996R import init_servos, Servo2Pos
from Hardware.Motores_DC.Desplazamiento import Carro
import time
import cv2
import numpy as np
import onnxruntime as ort  # Motor de inferencia ultraligero para RPi5
# =====================================================================


# --- Clase PID robusta--- 
class PID:
    def __init__(self,kP,kI,kD):
        self.kP,self.kI,self.kD = kP,kI,kD
        self.last_error = 0
        self.integral = 0 
        self.last_time = time.time()
    
    def update(self,error):
        now = time.time()
        dt = now - self.last_time
        if dt  <= 0: return 0
        
        #Proporcional
        P = self.kP * error 

        #Integral
        self.integral += error * dt
        #Limitar la integral(antiwindup) para evitar los latigazos
        self.integral= max(-10,min(10,self.integral))
        I = self.kI * self.integral
        
        #Derivativo
        D = self.kD * (error - self.last_error) / dt

        self.last_error = error
        self.last_time = now
        return P + I + D

# ---- Parametros del caarror ----
pines_izq = (17,27,12)
pines_der = (23,22,13)
pin_stby = 24
#carrito = Carro(pines_izq, pines_der, stby_pin=pin_stby)


# =====================================================================
# MODIFICACIÓN INTELIGENTE: INICIALIZACIÓN DE U-NET, ARUCO Y CÁMARA
# =====================================================================
# 1. Inicializar sesión de ONNX Runtime para la U-Net separable entrenada
MODEL_PATH = "/home/pacon/Tesis_pacon/Jupyter/Tesis-Proyecto/data/segmentador_color_640.onnx"
ort_session = ort.InferenceSession(MODEL_PATH)
input_name = ort_session.get_inputs()[0].name

# 2. Configurar detector métrico ArUco (Diccionario DICT_4X4_50)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# 3. Constantes Físicas de la tarjeta de calibración (Parámetros de tu tesis)
ANCHO_ARUCO_CM = 5.0      # Tamaño real impreso del marcador negro
DISTANCIA_FIXED_CM = 7.2  # Constante matemática 'd' de centro a centro

# 4. Matrices de Calibración Óptica (Ritual del tablero de ajedrez)
# Reemplaza estos valores con las constantes exactas de tu cámara
camera_matrix = np.array([[650.0, 0.0, 320.0],
                          [0.0, 650.0, 240.0],
                          [0.0, 0.0, 1.0]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# 5. Paleta de Clases e Identificación
CLASE_OBJETIVO = 2  # Cambiar según objetivo: 1=Rojo, 2=Verde, 3=Azul
PALETA_COLORES = {
    1: (0, 0, 255),   # Rojo BGR
    2: (0, 255, 0),   # Verde BGR
    3: (255, 0, 0)    # Azul BGR
}

# Inicializar cámara
cam = init_cam()
# =====================================================================


# Servo (solo eje X)
servo_x, servo_y = init_servos()

#constante de resoluciones y centro
FRAME_W = 640
FRAME_H = 480

CENTER_X = FRAME_W // 2
CENTER_Y = FRAME_H//2

#Variables de estado del servo
start_angle = 90.0 #Float para mayor presicion en el ajuste
angle_x = 90
angle_y = 90

dead_zone = 5 #Banda muerta reducida gracias al PID

#Timeout de inercia ayuda a que si el recall falla,el servo no se detenga
last_detection_time = time.time()
detection_timeout = 0.2 #segundos

#Delimitacion de angulos para la camara
angle_x_limit = [45,135]

# --- Sincronización PID ----
pid_x = PID(kP=0.01,kI=0.02,kD=0.0005)
pid_y = PID(kP=0.01, kI=0.02, kD=0.0005)

print("Iniciando prueba en tiempo real...")
Servo2Pos(servo_x, int(start_angle))
Servo2Pos(servo_y, int(start_angle))
cv2.namedWindow('Comparacion', cv2.WINDOW_AUTOSIZE)

while True:

    frame = cam.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    h_orig, w_orig, _ = frame.shape

    # Crear frame duplicado limpio para la telemetría visual de la tesis
    display_frame = frame.copy()


    # =====================================================================
    # MODIFICACIÓN INTELIGENTE: DETECCIÓN METROLÓGICA (ARUCO)
    # =====================================================================
    # Procesar canal gris para buscar los bits del marcador ArUco
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    z_real_cm = None
    if ids is not None:
        # Dibujar contorno del ArUco e inyectar telemetría en el frame
        cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, ANCHO_ARUCO_CM, camera_matrix, dist_coeffs
        )
        tvec = tvecs[0][0]
        z_real_cm = tvec[2]  # Distancia física perpendicular en centímetros Z
        
        cv2.putText(display_frame, f"ArUco ID: {ids[0][0]} Z: {z_real_cm:.1f}cm", 
                    (int(corners[0][0][0][0]), int(corners[0][0][0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    # =====================================================================


    # =====================================================================
    # MODIFICACIÓN INTELIGENTE: INFERENCE PIPELINE U-NET MULTICLASE (ONNX)
    # =====================================================================
    # Normalización ImageNet estricta e idéntica al entrenamiento de Colab
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    
    # Transponer dimensiones a formato de tensor PyTorch [1, C, H, W]
    img_input = np.transpose(img_normalized, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)

    # Inferencia del grafo ONNX en la CPU de la Raspberry Pi 5
    outputs = ort_session.run(None, {input_name: img_input})
    pred_map = np.argmax(outputs[0], axis=1)[0]  # Matriz de clases de [640, 640]
    
    # Escalar el mapa probabilístico de vuelta a la resolución real de tu cámara
    pred_map_orig = cv2.resize(pred_map.astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    # =====================================================================


    # =====================================================================
    # MODIFICACIÓN INTELIGENTE: FUSIÓN SENSORIAL Y EXTRACCIÓN DE CENTROIDE
    # =====================================================================
    best_centroid = None
    mask_visual_bgr = np.zeros_like(frame)

    # Iterar sobre las clases cromáticas para pintar la predicción en pantalla
    for clase_id, bgr_color in PALETA_COLORES.items():
        clase_mask = (pred_map_orig == clase_id).astype(np.uint8) * 255
        mask_visual_bgr[clase_mask == 255] = bgr_color
        
        # Si corresponde al color que deseamos rastrear en este ciclo, calculamos centroide
        if clase_id == CLASE_OBJETIVO:
            M = cv2.moments(clase_mask)
            if M["m00"] > 800:  # Umbral de píxeles filtrado contra ruido residual
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best_centroid = (cx, cy)
                last_detection_time = time.time()
                
                # Inyectar transparencia estética sobre el frame original
                display_frame[clase_mask == 255] = cv2.addWeighted(
                    display_frame[clase_mask == 255], 0.5, np.array(bgr_color, dtype=np.uint8), 0.5, 0
                )
    # =====================================================================


    #4. Logica de control PID
    #Si detectctamos el objeto o si lo perdimos hace muy poco (inercia)
    if best_centroid  or (time.time() - last_detection_time < detection_timeout):
        
        #Si best_centroid es None (falla el recall), usamos el ultimo error conocido
        if best_centroid:
            error_x = CENTER_X - best_centroid[0] #Error relativo al centro 
            error_y = CENTER_Y - best_centroid[1]
        else:
            error_x = pid_x.last_error #Mantener direccion
            error_y = pid_y.last_error

        # --- CONTROL EJE X ---
        if abs(error_x) > dead_zone:
            adjustment_x = pid_x.update(error_x)
            angle_x += adjustment_x
            angle_x = max(10, min(170, angle_x))  
            Servo2Pos(servo_x, int(angle_x)) # <-- Se envía entero para el hardware

        # --- CONTROL EJE Y ---
        if abs(error_y) > dead_zone:
            adjustment_y = pid_y.update(error_y)
            angle_y += adjustment_y 
            angle_y = max(60, min(130, angle_y))  # <-- CORREGIDO: Filtrar con angle_y
            Servo2Pos(servo_y, int(angle_y))
        

        # =====================================================================
        # MODIFICACIÓN INTELIGENTE: TELEMETRÍA 3D METROLÓGICA (TESIS)
        # =====================================================================
        if best_centroid:
            cv2.circle(display_frame, best_centroid, 8, (255, 255, 255), -1)
            cv2.putText(display_frame, "IA TRACKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Si el ArUco de la tarjeta está visible, estimar la coordenada espacial real en cm
            if z_real_cm is not None:
                focal_length = camera_matrix[0, 0]
                x_real_cm = ((best_centroid[0] - camera_matrix[0, 2]) * z_real_cm) / focal_length
                y_real_cm = ((best_centroid[1] - camera_matrix[1, 2]) * z_real_cm) / focal_length
                
                texto_3d = f"XYZ_Fisico: ({x_real_cm:.1f}, {y_real_cm:.1f}, {z_real_cm:.1f}) cm"
                cv2.putText(display_frame, texto_3d, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                print(f"Coordenadas Objetivo -> {texto_3d}")
        # =====================================================================

    ''' # CONTROL DEL CHASIS SIEMPRE
        if angle_x < angle_x_limit[0]:
            carrito.girar_izquierda()

        elif angle_x > angle_x_limit[1]:
            carrito.girar_derecha()

        else:
            carrito.detener()
    '''

    # Visualizacion lado a lado (Frame con Telemetría + Máscara Multiclase colorizada de la IA)
    combined = np.hstack((display_frame, mask_visual_bgr))
    cv2.imshow("Comparacion", combined)

    if cv2.waitKey(1) & 0xFF == 13:
        break

# CIERRE
cam.stop()
cv2.destroyAllWindows()