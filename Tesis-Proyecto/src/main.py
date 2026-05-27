from Vision.detector import ObjectDetector, obtener_mask
from Vision.camara import init_cam
from Hardware.Servomotores.MG996R import init_servos, Servo2Pos
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

# --- Configuracion de archivos ---
detector = ObjectDetector("/home/pacon/Tesis_pacon/Jupyter/Tesis-Proyecto/data/model.pkl",
                          "/home/pacon/Tesis_pacon/Jupyter/Tesis-Proyecto/data/scaler.pkl")

cam = init_cam()

# Servo (solo eje X)
servo_x, _ = init_servos()

#constante de resoluciones y centro
FRAME_W = 640
CENTER_X = FRAME_W // 2

#Variables de estado del servo
angle_x = 90.0 #Float para mayor presicion en el ajuste
dead_zone = 5 #Banda muerta reducida gracias al PID

#Timeout de inercia ayuda a que si el recall falla,el servo no se detenga
last_detection_time = time.time()
detection_timeout = 0.2 #segundos

# --- Sincronización PID ----
#kP: Reaccion inicial, kD: Amortigua el temblor, kI: Presicion final
#Valores departida [0.06,,0.02,0.0005] ajustar a gusto
pid_x = PID(kP=0.01,kI=0.02,kD=0.0005)


# --- Rangos HSV rojo ----
low_red1,up_red1 = np.array([0, 120, 90]),np.array([10, 255, 255])
low_red2, up_red2 = np.array([170, 120, 90]),np.array([180, 255, 255])


print("Iniciando prueba en tiempo real...")
Servo2Pos(servo_x, int(angle_x))
cv2.namedWindow('Comparacion', cv2.WINDOW_AUTOSIZE)

while True:

    frame = cam.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    #1. Preprocesamiento de color (solo 1 vez)
    hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #2. Deteccion con el nuevo detector (incluye escalado y solidez internamente)
    mask1 = obtener_mask(hsv_frame, low_red1, up_red1)
    mask2 = obtener_mask(hsv_frame, low_red2, up_red2)
    final_mask = detector.process_frame(frame, cv2.add(mask1,mask2))
   
   # 3. Localización del centroide más grande
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_centroid = None

    if contours:
        c = max(contours,key = cv2.contourArea)
        if cv2.contourArea(c) > 600:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                best_centroid = (cx,cy)
                last_detection_time = time.time()

    #4. Logica de control PID
    #Si detectctamos el objeto o si lo perdimos hace muy poco (inercia)
    if best_centroid  or (time.time() - last_detection_time < detection_timeout):
        
        #Si best_centroid es None (falla el recall), usamos el ultimo error conocido
        if best_centroid:
            error_x = CENTER_X - best_centroid[0] #Error relativo al centro 
        else:
            error_x = pid_x.last_error #Mantener direccion

        if abs(error_x) > dead_zone:
            #El PID calcula cuanto movernos, no a donde ir directamente
            adjustment = pid_x.update(error_x)
            #Actualizamos la posicion actual (invertir el signo si el servo va al lado contrario)
            angle_x += adjustment

            #Limitar el rango fisico de movimiento
            angle_x = max(10,min(170,angle_x))
            Servo2Pos(servo_x,angle_x)
            print(angle_x)
  
    #Visualizacion
    display_frame = frame.copy()
    if best_centroid:
        cv2.circle(display_frame,best_centroid,10,(0,255,0),1)
        cv2.putText(display_frame,"siguiendo",(10,30),1,1,(0,255,0),2)

    ML_mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((display_frame, ML_mask_bgr))

    cv2.imshow("Comparacion", combined)

    if cv2.waitKey(1) & 0xFF == 13:
        break
# CIERRE

cam.stop()
cv2.destroyAllWindows()