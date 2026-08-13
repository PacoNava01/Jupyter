from Vision.camara import init_cam
from Hardware.Servomotores.MG996R import init_servos, Servo2Pos
from Hardware.Motores_DC.Desplazamiento import Carro
import time
import cv2
import numpy as np
from ultralytics import YOLO

# =====================================================================
# 1. CLASE CONTROLADOR PID ROBUSTO
# =====================================================================
class PID:
    def __init__(self, kP, kI, kD):
        self.kP, self.kI, self.kD = kP, kI, kD
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0: return 0
        
        P = self.kP * error
        self.integral = max(-10, min(10, self.integral + error * dt)) # Anti-windup
        I = self.kI * self.integral
        D = self.kD * (error - self.last_error) / dt
        
        self.last_error = error
        self.last_time = now
        return P + I + D

# ---- Parámetros del coche ----
pines_izq = (17, 27, 12)
pines_der = (23, 22, 13)
pin_stby = 24
# carrito = Carro(pines_izq, pines_der, stby_pin=pin_stby)

# =====================================================================
# 2. CONFIGURACIÓN DE IA (YOLOv8), ARUCO Y SISTEMA ÓPTICO
# =====================================================================
# Cargar modelo YOLOv8 entrenado (Ruta en tu Pi)
MODEL_PATH = "/home/pacon/Jupyter/Tesis-Proyecto/data/yolov8n_colores_best.onnx"
model = YOLO(MODEL_PATH, task='segment')

# Configurar Detector ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
ANCHO_ARUCO_CM = 5.0

# Matriz de Calibración de la cámara (Reemplazar con tus coeficientes exactos)
camera_matrix = np.array([[650.0, 0.0, 320.0], 
                          [0.0, 650.0, 240.0], 
                          [0.0, 0.0, 1.0]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# SELECCIÓN DEL COLOR OBJETIVO: 'rojo', 'verde', o 'azul'
CLASE_OBJETIVO = "rojo" 

# Configuración de Hardware
cam = init_cam()
servo_x, servo_y = init_servos()

FRAME_W, FRAME_H = 640, 480
CENTER_X = FRAME_W // 2
CENTER_Y = FRAME_H // 2

start_angle = 90.0
angle_x, angle_y = 90, 90
dead_zone = 5
last_detection_time = time.time()
detection_timeout = 0.3

# Inicializar Controladores PID
pid_x = PID(kP=0.01, kI=0.02, kD=0.0005)
pid_y = PID(kP=0.01, kI=0.02, kD=0.0005)

# Posición inicial de reposo
Servo2Pos(servo_x, int(angle_x))
Servo2Pos(servo_y, int(angle_y))

print(f"🚀 Iniciando seguimiento en tiempo real con YOLOv8-seg para el objetivo: [{CLASE_OBJETIVO.upper()}]...")
cv2.namedWindow("Visión Robot YOLOv8", cv2.WINDOW_AUTOSIZE)

# =====================================================================
# 3. BUCLE PRINCIPAL DE NAVEGACIÓN Y SEGUIMIENTO
# =====================================================================
while True:
    frame = cam.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    display_frame = frame.copy()

    # -----------------------------------------------------------------
    # A. DETECCIÓN METROLÓGICA (ARUCO)
    # -----------------------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    z_real_cm = None

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, ANCHO_ARUCO_CM, camera_matrix, dist_coeffs)
        z_real_cm = tvecs[0][0][2] # Distancia Z en centímetros
        cv2.putText(display_frame, f"ArUco Z: {z_real_cm:.1f}cm", 
                    (int(corners[0][0][0][0]), int(corners[0][0][0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # -----------------------------------------------------------------
    # B. INFERENCIA Y SEGMENTACIÓN CROMÁTICA CON YOLOV8
    # -----------------------------------------------------------------
    # Inferencia rápida (umbral de confianza del 40%)
    results = model(frame, verbose=False, conf=0.4)[0]
    best_centroid = None

    if results.masks is not None:
        for mask, box in zip(results.masks.xy, results.boxes):
            cls_id = int(box.cls[0])
            label_name = model.names[cls_id]

            # Verificar si la figura detectada coincide con nuestro color objetivo
            if label_name == CLASE_OBJETIVO and len(mask) > 0:
                pts = np.int32([mask])
                
                # Dibujar contorno exterior de la figura segmentada
                cv2.polylines(display_frame, pts, True, (0, 255, 0), 2)

                # Calcular centroide geométrico por momentos
                M = cv2.moments(pts)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    best_centroid = (cx, cy)
                    last_detection_time = time.time()
                    break

    # -----------------------------------------------------------------
    # C. CONTROL PID DE SERVOMOTORES Y TELEMETRÍA 3D
    # -----------------------------------------------------------------
    if best_centroid or (time.time() - last_detection_time < detection_timeout):
        if best_centroid:
            error_x = CENTER_X - best_centroid[0]
            error_y = CENTER_Y - best_centroid[1]
        else:
            error_x = pid_x.last_error
            error_y = pid_y.last_error

        # Actualizar Servomotor Eje X
        if abs(error_x) > dead_zone:
            angle_x = max(10, min(170, angle_x + pid_x.update(error_x)))
            Servo2Pos(servo_x, int(angle_x))

        # Actualizar Servomotor Eje Y
        if abs(error_y) > dead_zone:
            angle_y = max(60, min(130, angle_y + pid_y.update(error_y)))
            Servo2Pos(servo_y, int(angle_y))

        # Inyectar información visual en pantalla
        if best_centroid:
            cv2.circle(display_frame, best_centroid, 6, (255, 255, 255), -1)
            cv2.putText(display_frame, f"TRACKING [{CLASE_OBJETIVO.upper()}]", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Si el marcador ArUco está visible, calculamos posición 3D real
            if z_real_cm is not None:
                fx = camera_matrix[0, 0]
                fy = camera_matrix[1, 1]
                x_cm = ((best_centroid[0] - camera_matrix[0, 2]) * z_real_cm) / fx
                y_cm = ((best_centroid[1] - camera_matrix[1, 2]) * z_real_cm) / fy
                
                texto_3d = f"Posicion Real: X={x_cm:.1f}cm, Y={y_cm:.1f}cm, Z={z_real_cm:.1f}cm"
                cv2.putText(display_frame, texto_3d, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Renderizar frame con telemetría en vivo
    cv2.imshow("Visión Robot YOLOv8", display_frame)

    if cv2.waitKey(1) & 0xFF == 13: # Presionar Enter para salir
        break

# CIERRE SEGURO DE HARDWARE
cam.stop()
cv2.destroyAllWindows()