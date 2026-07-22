import time
import cv2
import numpy as np

# =====================================================================
# MODIFICACIÓN INTELIGENTE: BINDINGS OFICIALES DE HAILORT (ULTRA-RÁPIDOS)
# =====================================================================
from Vision.camara import init_cam
from Hardware.Servomotores.MG996R import init_servos, Servo2Pos
from Hardware.Motores_DC.Desplazamiento import Carro

from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    InferVStreams,
    FormatType
)
# =====================================================================


# --- Clase PID robusta --- 
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
        self.integral += error * dt
        self.integral = max(-10, min(10, self.integral))
        I = self.kI * self.integral
        D = self.kD * (error - self.last_error) / dt

        self.last_error = error
        self.last_time = now
        return P + I + D


# ---- Parametros del carro ----
pines_izq = (17, 27, 12)
pines_der = (23, 22, 13)
pin_stby = 24
# carrito = Carro(pines_izq, pines_der, stby_pin=pin_stby)


# =====================================================================
# MODIFICACIÓN INTELIGENTE: INICIALIZACIÓN CON UINT8 (CERO CARGA CPU)
# =====================================================================
MODEL_PATH = "/home/pacon/Tesis_pacon/Jupyter/Tesis-Proyecto/data/segmentador_color_640.hef"

hef = HEF(MODEL_PATH)
target = VDevice()

configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
network_groups = target.configure(hef, configure_params)
network_group = network_groups[0]
network_group_params = network_group.create_params()

# OTROS PARÁMETROS CRÍTICOS: UINT8 permite enviar frames sin conversión float
input_vstream_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
output_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.UINT8)

input_vstream_info = hef.get_input_vstream_infos()[0]
output_vstream_info = hef.get_output_vstream_infos()[0]

# Configurar detector ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Constantes Físicas y Ópticas
ANCHO_ARUCO_CM = 5.0      
camera_matrix = np.array([[650.0, 0.0, 320.0],
                          [0.0, 650.0, 240.0],
                          [0.0, 0.0, 1.0]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

CLASE_OBJETIVO = 2  # 1=Rojo, 2=Verde, 3=Azul
COLOR_OBJETIVO_BGR = (0, 255, 0) # Verde para contorno

cam = init_cam()
# =====================================================================


# Servos
servo_x, servo_y = init_servos()

FRAME_W = 640
FRAME_H = 480

CENTER_X = FRAME_W // 2
CENTER_Y = FRAME_H // 2

start_angle = 90.0 
angle_x = 90
angle_y = 90

dead_zone = 5 
last_detection_time = time.time()
detection_timeout = 0.2 

pid_x = PID(kP=0.01, kI=0.02, kD=0.0005)
pid_y = PID(kP=0.01, kI=0.02, kD=0.0005)

print("Iniciando prueba optimizada en tiempo real (Modo Ultra-Baja Latencia)...")
Servo2Pos(servo_x, int(start_angle))
Servo2Pos(servo_y, int(start_angle))
cv2.namedWindow('Visión Robot', cv2.WINDOW_AUTOSIZE)


# =====================================================================
# MODIFICACIÓN INTELIGENTE: BUCLE DE ULTRA-BAJA LATENCIA (HAILO NPU)
# =====================================================================
with network_group.activate(network_group_params):
    with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
        
        while True:
            frame = cam.capture_array()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            h_orig, w_orig, _ = frame.shape

            # ---------------------------------------------------------
            # 1. RASTREO METROLÓGICO ARUCO (Optimizado)
            # ---------------------------------------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco_detector.detectMarkers(gray)

            z_real_cm = None
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, ANCHO_ARUCO_CM, camera_matrix, dist_coeffs
                )
                z_real_cm = tvecs[0][0][2]  
                cv2.putText(frame, f"Z: {z_real_cm:.1f}cm", 
                            (int(corners[0][0][0][0]), int(corners[0][0][0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # ---------------------------------------------------------
            # 2. INFERENCIA DIRECTA EN NPU (Sin preprocesamiento CPU)
            # ---------------------------------------------------------
            # Ajuste de tamaño simple en enteros UINT8 sin divisiones flotantes
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (640, 640))
            img_input = np.expand_dims(img_resized, axis=0) # [1, 640, 640, 3] en UINT8

            # Inferencia PCIe en milisegundos
            raw_results = infer_pipeline.infer({input_vstream_info.name: img_input})
            output_tensor = raw_results[output_vstream_info.name]

            # Extraer mapa de predicción
            if output_tensor.ndim == 4 and output_tensor.shape[1] == 4:
                pred_map = np.argmax(output_tensor[0], axis=0)
            else:
                pred_map = np.argmax(output_tensor[0], axis=-1)

            # ---------------------------------------------------------
            # 3. CENTROIDE RÁPIDO SIN RE-ESCALAR TODA LA IMAGEN
            # ---------------------------------------------------------
            # Aislar la clase deseada directamente en la escala 640x640 de la NPU
            clase_mask_small = (pred_map == CLASE_OBJETIVO).astype(np.uint8) * 255
            
            best_centroid = None
            M = cv2.moments(clase_mask_small)
            if M["m00"] > 400:  # Umbral ajustado a escala 640x640
                # Coordenadas en la escala de la red
                cx_small = int(M["m10"] / M["m00"])
                cy_small = int(M["m01"] / M["m00"])
                
                # Mapear coordenadas al tamaño real del marco de la cámara (Escalado instantáneo)
                cx = int(cx_small * (w_orig / 640.0))
                cy = int(cy_small * (h_orig / 640.0))
                best_centroid = (cx, cy)
                last_detection_time = time.time()

                # Dibujar contorno ligero en lugar de pintar toda la máscara pixel a pixel
                contours, _ = cv2.findContours(clase_mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c_max = max(contours, key=cv2.contourArea)
                    # Escalar contorno para el frame visual
                    c_scaled = (c_max * [w_orig / 640.0, h_orig / 640.0]).astype(np.int32)
                    cv2.drawContours(frame, [c_scaled], -1, COLOR_OBJETIVO_BGR, 2)

            # ---------------------------------------------------------
            # 4. LÓGICA DE CONTROL PID
            # ---------------------------------------------------------
            if best_centroid or (time.time() - last_detection_time < detection_timeout):
                if best_centroid:
                    error_x = CENTER_X - best_centroid[0] 
                    error_y = CENTER_Y - best_centroid[1]
                else:
                    error_x = pid_x.last_error 
                    error_y = pid_y.last_error

                # Control X
                if abs(error_x) > dead_zone:
                    angle_x = max(10, min(170, angle_x + pid_x.update(error_x)))  
                    Servo2Pos(servo_x, int(angle_x)) 

                # Control Y
                if abs(error_y) > dead_zone:
                    angle_y = max(60, min(130, angle_y + pid_y.update(error_y)))  
                    Servo2Pos(servo_y, int(angle_y))
                
                # Telemetría en pantalla
                if best_centroid:
                    cv2.circle(frame, best_centroid, 6, (255, 255, 255), -1)
                    cv2.putText(frame, "HAILO-8L HIGH-FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if z_real_cm is not None:
                        focal_length = camera_matrix[0, 0]
                        x_real_cm = ((best_centroid[0] - camera_matrix[0, 2]) * z_real_cm) / focal_length
                        y_real_cm = ((best_centroid[1] - camera_matrix[1, 2]) * z_real_cm) / focal_length
                        
                        cv2.putText(frame, f"XYZ: ({x_real_cm:.1f}, {y_real_cm:.1f}, {z_real_cm:.1f}) cm", 
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # ---------------------------------------------------------
            # 5. MOSTRAR UN SOLO FRAME LIMPIO (Sin np.hstack)
            # ---------------------------------------------------------
            cv2.imshow("Visión Robot", frame)

            if cv2.waitKey(1) & 0xFF == 13:
                break
# =====================================================================

# CIERRE
cam.stop()
cv2.destroyAllWindows()