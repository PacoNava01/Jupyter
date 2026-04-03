import cv2
import numpy as np
from picamera2 import Picamera2

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

        if centroide:
            cv2.putText(display_frame, f"Pos: {centroide}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 5. Mostrar resultados
        cv2.imshow("Deteccion", display_frame)
        cv2.imshow("Mascara", mask_red)

        if cv2.waitKey(1) & 0xFF == 13: # Enter para salir
            break

finally:
    if 'camara' in locals():
        camara.stop()
    cv2.destroyAllWindows()