import cv2
import numpy as np
from picamera2 import Picamera2
import time

'''
Definiremos un contro PID que nos permita controlar de manera 
más eficaz e "inteligente
'''
class PID:
    def __init__(self,kP,kI,kD):
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.last_error = 0
        self.integral = 0
        self.las_time = time.time()
    
    def update(self,error):
        now = time.time()
        dt = now - self.las_time()
        if dt <= 0:
            return 0
    
        #Proporcional
        P = self.kP * error
        #Integral
        self.integral += error * dt
        I = self.kI * self.integral
        #Derivativo
        D = self.kD * (error-self.last_error)/dt

        self.last_error = error
        self.las_time = now

        return P + I + D





def init_cam():
    try:
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"})
        cam.configure(config)
        cam.start()
        print("Cámara inicializada correctamente.")
        return cam
    except Exception as e:
        print(f"Error al inicializar la cámara: {e}")
        return None

def obtener_mask(frame_rgb, low_hsv, up_hsv):
    # IMPORTANTE: aunque la camara captura en RGB, OpenCV Lee BGR, por eso convertimos a HSV
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_hsv, up_hsv)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def procesar_contornos(mask, frame_para_dibujar, min_area):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroide_detectado = None
    
    if contornos:
        contornos_validos = [c for c in contornos if cv2.contourArea(c) > min_area]
        if contornos_validos:
            c = max(contornos_validos, key=cv2.contourArea) #Tomamos el más grande
            x, y, w, h = cv2.boundingRect(c)
            
            # Dibujamos sobre la copia (frame_para_dibujar ya debe estar en BGR)
            cv2.rectangle(frame_para_dibujar, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame_para_dibujar, (cx, cy), 5, (0, 0, 255), -1)
                centroide_detectado = (cx, cy)
                
    return frame_para_dibujar, centroide_detectado

# --- Rangos Rojos (Ajustados para RGB -> HSV) ---
low_red1 = np.array([0, 100, 50], dtype=np.uint8)
up_red1 = np.array([10, 255, 255], dtype=np.uint8)
low_red2 = np.array([170, 100, 50], dtype=np.uint8)
up_red2 = np.array([180, 255, 255], dtype=np.uint8)


# --- Configuracion inicial  del PID en x y y---
pid_x = PID(kP = 0.1, kI = 0.01,kD = 0.005)
pid_y = PID(kP = 0.1, kI = 0.01,kD = 0.005)

centro_pantalla_x = 320
centro_pantalla_y = 240
posicion_servo = 90 #Empezamos en el centro de nuestro rango

try:
    camara = init_cam()
    if camara is None: exit()
    
    cv2.namedWindow("Deteccion")
    cv2.namedWindow("Mascara")

    while True:
        
        frame_raw = camara.capture_array()
        if frame_raw is None: break

        # 3. Crear máscara (Usamos el raw que es RGB)
        mask1 = obtener_mask(frame_raw, low_red1, up_red1)
        mask2 = obtener_mask(frame_raw, low_red2, up_red2)
        mask_red = cv2.add(mask1, mask2)
    
        # 4. Procesar y dibujar en la copia de visualización
        display_frame = frame_raw.copy()  # Hacemos una copia AQUÍ para que los dibujos no afecten capturas futuras
        display_frame, centroide = procesar_contornos(mask_red, display_frame, min_area=650)

        if centroide is not None:
            cv2.putText(display_frame, f"Pos: {centroide}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cx,cy = centroide
            error_x = centro_pantalla_x - cx
            error_y = centro_pantalla_x - cy

            #Obtenemos el ajuste del PID
            ajuste_x =  pid_x.update(error_x)
            ajuste_y =  pid_y.update(error_y)


            #Actualizar la posicion del servo
            posicion_servo =+ ajuste_x
            posicion_servo =+ ajuste_y

            #Limitar para que el servo no intente ir màs alla de sus limites
            posicion_servo_x = max(0,min(180,posicion_servo_x))
            posicion_servo_y = max(0,min(180,posicion_servo_y))

            #Mandamos posiciones a servos


        # 5. Mostrar resultados
        cv2.imshow("Deteccion", display_frame)
        cv2.imshow("Mascara", mask_red)

        if cv2.waitKey(1) & 0xFF == 13: # Enter para salir
            break

finally:
    if 'camara' in locals():
        camara.stop()
    cv2.destroyAllWindows()