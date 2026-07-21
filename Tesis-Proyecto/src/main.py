import time
import cv2
import numpy as np

# =====================================================================
# MODIFICACIÓN INTELIGENTE: BINDINGS OFICIALES DE HAILORT
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
        
        # Proporcional
        P = self.kP * error 

        # Integral
        self.integral += error * dt
        self.integral = max(-10, min(10, self.integral))
        I = self.kI * self.integral
        
        # Derivativo
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
# MODIFICACIÓN INTELIGENTE: CONFIGURACIÓN E INICIALIZACIÓN DEL HAT HAILO-8L
# =====================================================================
MODEL_PATH = "/home/pacon/Tesis_pacon/Jupyter/Tesis-Proyecto/data/segmentador_color_640.hef"

# 1. Cargar el modelo compilado HEF
hef = HEF(MODEL_PATH)

# 2. Inicializar el dispositivo PCI-e (VDevice detecta el nodo /dev/hailo0)
target = VDevice()

# 3. Configurar el grupo de red en el hardware
configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
network_groups = target.configure(hef, configure_params)
network_group = network_groups[0]
network_group_params = network_group.create_params()

# 4. Definir los parámetros de transmisión de entrada/salida para baja latencia
input_vstream_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
output_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

# Extraer los nombres dinámicos de los streams
input_vstream_info = hef.get_input_vstream_infos()[0]
output_vstream_info = hef.get_output_vstream_infos()[0]

# 5. Configurar detector métrico ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# 6. Constantes Físicas de la tarjeta
ANCHO_ARUCO_CM = 5.0      
DISTANCIA_FIXED_CM = 7.2  

# 7. Matrices de Calibración Óptica
camera_matrix = np.array([[650.0, 0.0, 320.0],
                          [0.0, 650.0, 240.0],
                          [0.0, 0.0, 1.0]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# 8. Paleta de Clases
CLASE_OBJETIVO = 2  # 1=Rojo, 2=Verde, 3=Azul
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

# Constante de resoluciones y centro
FRAME_W = 640
FRAME_H = 480

CENTER_X = FRAME_W // 2
CENTER_Y = FRAME_H // 2

# Variables de estado del servo
start_angle = 90.0 
angle_x = 90
angle_y = 90

dead_zone = 5 

last_detection_time = time.time()
detection_timeout = 0.2 

angle_x_limit = [45, 135]

# --- Sincronización PID ----
pid_x = PID(kP=0.01, kI=0.02, kD=0.0005)
pid_y = PID(kP=0.01, kI=0.02, kD=0.0005)

print("Iniciando prueba en tiempo real con Aceleración NPU Hailo-8L...")
Servo2Pos(servo_x, int(start_angle))
Servo2Pos(servo_y, int(start_angle))
cv2.namedWindow('Comparacion', cv2.WINDOW_AUTOSIZE)


# =====================================================================
# MODIFICACIÓN INTELIGENTE: PIPELINE DE INFERENCIA CONTINUA EN EL CHIP
# =====================================================================
# Activamos el chip NPU y creamos el pipeline de streaming antes de entrar al bucle principal
with network_group.activate(network_group_params):
    with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
        
        while True:
            frame = cam.capture_array()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            h_orig, w_orig, _ = frame.shape

            display_frame = frame.copy()

            # --- RASTREO METROLÓGICO (ARUCO) ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco_detector.detectMarkers(gray)

            z_real_cm = None
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, ANCHO_ARUCO_CM, camera_matrix, dist_coeffs
                )
                tvec = tvecs[0][0]
                z_real_cm = tvec[2]  
                
                cv2.putText(display_frame, f"ArUco ID: {ids[0][0]} Z: {z_real_cm:.1f}cm", 
                            (int(corners[0][0][0][0]), int(corners[0][0][0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # --- PREPROCESAMIENTO PARA CHIP HAILO ---
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (640, 640))
            
            # Normalización estándar
            img_normalized = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_normalized = (img_normalized - mean) / std
            
            # Hailo NPU procesa en formato NHWC [Batch, H, W, C]
            img_input = np.expand_dims(img_normalized, axis=0).astype(np.float32)

            # --- INFERENCIA ULTRA RÁPIDA EN EL NPU ---
            # Se envía el marco directamente al bus PCIe hacia el chip Hailo
            input_data = {input_vstream_info.name: img_input}
            raw_results = infer_pipeline.infer(input_data)
            output_tensor = raw_results[output_vstream_info.name]

            # Procesar salida probabilística [1, H, W, Clases] o [1, Clases, H, W]
            if output_tensor.ndim == 4 and output_tensor.shape[1] == 4:
                pred_map = np.argmax(output_tensor[0], axis=0) # Formato NCHW
            else:
                pred_map = np.argmax(output_tensor[0], axis=-1) # Formato NHWC
            
            pred_map_orig = cv2.resize(pred_map.astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

            # --- FUSIÓN SENSORIAL Y EXTRACCIÓN DE CENTROIDE ---
            best_centroid = None
            mask_visual_bgr = np.zeros_like(frame)

            for clase_id, bgr_color in PALETA_COLORES.items():
                clase_mask = (pred_map_orig == clase_id).astype(np.uint8) * 255
                mask_visual_bgr[clase_mask == 255] = bgr_color
                
                if clase_id == CLASE_OBJETIVO:
                    M = cv2.moments(clase_mask)
                    if M["m00"] > 800:  
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        best_centroid = (cx, cy)
                        last_detection_time = time.time()
                        
                        display_frame[clase_mask == 255] = cv2.addWeighted(
                            display_frame[clase_mask == 255], 0.5, np.array(bgr_color, dtype=np.uint8), 0.5, 0
                        )

            # --- LÓGICA DE CONTROL PID ---
            if best_centroid or (time.time() - last_detection_time < detection_timeout):
                if best_centroid:
                    error_x = CENTER_X - best_centroid[0] 
                    error_y = CENTER_Y - best_centroid[1]
                else:
                    error_x = pid_x.last_error 
                    error_y = pid_y.last_error

                # CONTROL EJE X
                if abs(error_x) > dead_zone:
                    adjustment_x = pid_x.update(error_x)
                    angle_x += adjustment_x
                    angle_x = max(10, min(170, angle_x))  
                    Servo2Pos(servo_x, int(angle_x)) 

                # CONTROL EJE Y
                if abs(error_y) > dead_zone:
                    adjustment_y = pid_y.update(error_y)
                    angle_y += adjustment_y 
                    angle_y = max(60, min(130, angle_y))  
                    Servo2Pos(servo_y, int(angle_y))
                
                # TELEMETRÍA 3D
                if best_centroid:
                    cv2.circle(display_frame, best_centroid, 8, (255, 255, 255), -1)
                    cv2.putText(display_frame, "HAILO-8L TRACKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if z_real_cm is not None:
                        focal_length = camera_matrix[0, 0]
                        x_real_cm = ((best_centroid[0] - camera_matrix[0, 2]) * z_real_cm) / focal_length
                        y_real_cm = ((best_centroid[1] - camera_matrix[1, 2]) * z_real_cm) / focal_length
                        
                        texto_3d = f"XYZ_Fisico: ({x_real_cm:.1f}, {y_real_cm:.1f}, {z_real_cm:.1f}) cm"
                        cv2.putText(display_frame, texto_3d, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # VISUALIZACIÓN EN TIEMPO REAL
            combined = np.hstack((display_frame, mask_visual_bgr))
            cv2.imshow("Comparacion", combined)

            if cv2.waitKey(1) & 0xFF == 13:
                break
# =====================================================================

# CIERRE
cam.stop()
cv2.destroyAllWindows()