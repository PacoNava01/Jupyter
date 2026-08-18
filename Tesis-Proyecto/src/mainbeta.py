from Vision.detector import ObjectDetector, obtener_mask
from Vision.camara import init_cam
from Hardware.Servomotores.MG996R import init_servos, Servo2Pos
from Hardware.Motores_DC.Desplazamiento import Carro
import time
import cv2
import numpy as np

# --- Clase PID robusta --- 
class PID:
    def __init__(self,kP,kI,kD):
        self.kP,self.kI,self.kD = kP,kI,kD
        self.last_error = 0
        self.integral = 0 
        self.last_time = time.time()
    
    def update(self,error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0: return 0
        
        # Proporcional
        P = self.kP * error 

        # Integral (antiwindup)
        self.integral += error * dt
        self.integral = max(-10, min(10, self.integral))
        I = self.kI * self.integral
        
        # Derivativo
        D = self.kD * (error - self.last_error) / dt

        self.last_error = error
        self.last_time = now
        return P + I + D

# ---- Parámetros del carrito ----
pines_izq = (17,27,12)
pines_der = (23,22,13)
pin_stby = 24
carrito = Carro(pines_izq, pines_der, pin_stby)

# --- Configuración de archivos ---
detector = ObjectDetector(
    "/home/pacon/Tesis_pacon/Jupyter/Tesis-Proyecto/data/model.pkl",
    "/home/pacon/Tesis_pacon/Jupyter/Tesis-Proyecto/data/scaler.pkl"
)

cam = init_cam()

# Servo (solo eje X/Y)
servo_x, servo_y = init_servos()

# Constantes de resolución y centro
FRAME_W, FRAME_H = 640, 480
CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2

# Variables de estado del servo
start_angle = 90.0
angle_x, angle_y = 90, 90
dead_zone = 5
last_detection_time = time.time()
detection_timeout = 0.2

# Delimitación de ángulos para la cámara
angle_x_limit = [45,135]

# --- PID ---
pid_x = PID(kP=0.01, kI=0.02, kD=0.0005)
pid_y = PID(kP=0.01, kI=0.02, kD=0.0005)

# --- Rangos HSV rojo ---
low_red1 = np.array([0, 115, 20])
up_red1  = np.array([10, 255, 255])
low_red2 = np.array([170, 100, 35])
up_red2  = np.array([180, 255, 255])

print("Iniciando prueba en tiempo real...")
Servo2Pos(servo_x, int(start_angle))
Servo2Pos(servo_y, int(start_angle))
cv2.namedWindow('Comparacion', cv2.WINDOW_AUTOSIZE)

while True:
    frame = cam.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = obtener_mask(hsv_frame, low_red1, up_red1)
    mask2 = obtener_mask(hsv_frame, low_red2, up_red2)
    final_mask = detector.process_frame(hsv_frame, cv2.add(mask1,mask2))

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_centroid, area = None, 0

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > 800:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                best_centroid = (cx,cy)
                last_detection_time = time.time()

    if best_centroid or (time.time() - last_detection_time < detection_timeout):
        if best_centroid:
            error_x = CENTER_X - best_centroid[0]
            error_y = CENTER_Y - best_centroid[1]
        else:
            error_x = pid_x.last_error
            error_y = pid_y.last_error

        # --- CONTROL EJE X ---
        if abs(error_x) > dead_zone:
            adjustment_x = pid_x.update(error_x)
            angle_x += adjustment_x
            angle_x = max(10, min(170, angle_x))
            Servo2Pos(servo_x, int(angle_x))

        # --- CONTROL EJE Y ---
        if abs(error_y) > dead_zone:
            adjustment_y = pid_y.update(error_y)
            angle_y += adjustment_y 
            angle_y = max(60, min(130, angle_y))
            Servo2Pos(servo_y, int(angle_y))
        
        print(f"Cam: ({int(angle_x)}, {int(angle_y)})")

        # --- CONTROL DEL CHASIS ---
        if angle_x < angle_x_limit[0]:
            carrito.accion("izquierda")
        elif angle_x > angle_x_limit[1]:
            carrito.accion("derecha")
        else:
            # Avance/retroceso según área del objeto
            if area < 2000:
                carrito.accion("avanzar", velocidad=0.4)
            elif area > 5000:
                carrito.accion("retroceder", velocidad=0.3)
            else:
                carrito.detener()

    # Visualización
    display_frame = frame.copy()
    if best_centroid:
        cv2.circle(display_frame, best_centroid, 10, (0,255,0), 1)
        cv2.putText(display_frame,"siguiendo",(10,30),1,1,(0,255,0),2)
        cv2.putText(display_frame,f"{area}",(10,60),1,1,(0,255,0),2)

    ML_mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((display_frame, ML_mask_bgr))
    cv2.imshow("Comparacion", combined)

    if cv2.waitKey(1) & 0xFF == 13:
        break

# CIERRE
carrito.detener()
cam.stop()
cv2.destroyAllWindows()
