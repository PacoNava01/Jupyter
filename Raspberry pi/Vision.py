#Librerias de vision
import cv2
import numpy as np
from picamera2 import Picamera2
#-------------CONFIGURACIÓN CÁMARA-------------
def init_camara():
    """
    Inicializa y configura la cámara.
    Devuelve el objeto cámara si todo funciona,
    o None si ocurre algún error.
    """
    try:
        picam2 = Picamera2()

        config = picam2.create_video_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )

        picam2.configure(config)
        picam2.start()


        frame = picam2.capture_array()
        if frame is None:
            raise Exception("No se pudo capturar frame inicial.")

        print("Cámara inicializada correctamente.")
        return picam2

    except Exception as e:
        print(f"Error al inicializar cámara: {e}")
        return None

def capture_frame(camara):
    """
    Captura un frame de la cámara.
    """
    return camara.capture_array()

def mostrar_frame(nombre, frame):
    """
    Solo muestra el frame.
    La ventana debe crearse UNA sola vez fuera del loop.
    """
    cv2.imshow(nombre, frame)

#-------------PROCESAMIENTO DE IMAGEN-------------
def obtener_mask(frame_bgr, low_hsv, up_hsv):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_hsv, up_hsv)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def obtener_contornos(mask, frame_para_dibujar, min_area):
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
                
    return frame_para_dibujar, centroide_detectado,w

def Estimacion_focal_px(ancho_real, ancho_bbox, distancia_real):
    '''
    calcula la distancia focal en píxeles usando la fórmula:
    focal_px = (ancho * distancia_real) / ancho_bbox
    '''
    if ancho_bbox == 0:
        return None
    focal_px = (ancho_real * distancia_real) / ancho_bbox
    return focal_px

# --- Creamos trackbars para ajustar los rangos HSV en tiempo real ---
def nothing(x):
    pass


def crear_trackbar(nombre_ventana):
    cv2.createTrackbar("Low H", nombre_ventana, 0, 179, nothing)
    cv2.createTrackbar("Low S", nombre_ventana, 0, 255, nothing)
    cv2.createTrackbar("Low V", nombre_ventana, 0, 255, nothing)
    cv2.createTrackbar("Up H", nombre_ventana, 0, 179, nothing)
    cv2.createTrackbar("Up S", nombre_ventana, 0, 255, nothing)
    cv2.createTrackbar("Up V", nombre_ventana, 0, 255, nothing)

def trackbar_val(nombre_ventana):
    low_h = cv2.getTrackbarPos("Low H", nombre_ventana)
    low_s = cv2.getTrackbarPos("Low S", nombre_ventana)
    low_v = cv2.getTrackbarPos("Low V", nombre_ventana)
    up_h = cv2.getTrackbarPos("Up H", nombre_ventana)
    up_s = cv2.getTrackbarPos("Up S", nombre_ventana)
    up_v = cv2.getTrackbarPos("Up V", nombre_ventana)

    low = np.array([low_h, low_s, low_v], dtype=np.uint8)
    up = np.array([up_h, up_s, up_v], dtype=np.uint8)
    return low, up

def bgr_a_hsv(b, g, r):
    color_bgr = np.uint8([[[b, g, r]]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    return color_hsv[0][0]


def detectar_color(frame_bgr, color_bgr, thres_color=10):
    '''
    Detecta un color BGR específico usando un margen de tolerancia.
    Maneja automáticamente el caso especial del color rojo.
    '''
    #Convertimos el color objetivo a HSV
    hsv_objetivo = bgr_a_hsv(color_bgr[0], color_bgr[1], color_bgr[2])
    h_centro = hsv_objetivo[0]
    
    #Convertimos el frame a HSV una sola vez para ahorrar CPU
    hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2HSV)

    # CASO ESPECIAL: ROJO 
    # El rojo está cerca de 0 O cerca de 180
    if h_centro < 10 or h_centro > 170:
        # Rango 1: Desde 0 hacia arriba
        low1 = np.array([0, 100, 50])
        up1 = np.array([h_centro + thres_color, 255, 255])
        
        # Rango 2: Desde el final (179) hacia abajo
        low2 = np.array([179 - thres_color, 100, 50])
        up2 = np.array([179, 255, 255])
        
        mask1 = cv2.inRange(hsv_frame, low1, up1)
        mask2 = cv2.inRange(hsv_frame, low2, up2)
        mask = cv2.bitwise_or(mask1, mask2)
        
    # 4. CASO NORMAL: Cualquier otro color
    else:
        # Usamos clip para no salirnos de 0-179
        low = np.array([max(0, h_centro - thres_color), 100,100])
        up = np.array([min(179, h_centro + thres_color), 255, 255])
        mask = cv2.inRange(hsv_frame, low, up)
    
    # 5. Limpieza morfológica 
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def aplicar_mascara(frame_bgr, mask):
    """
    Aplica la máscara binaria sobre el frame original para mostrar 
    solo los píxeles detectados en sus colores reales.
    """
    # Bitwise_and: (Imagen A, Imagen B, sobre qué máscara)
    resultado = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
    return resultado


# --- Parametro ---
thres_color = 10


if __name__ == "__main__":
    camara = init_camara()
    if camara is None: exit()
    
    cv2.namedWindow("Deteccion",cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Mascara",cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Resultados",cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("controles")

    #crear_trackbar("controles")
    while True:
        frame_bgr = capture_frame(camara)
        if frame_bgr is None: break
        gaussian_blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        mask = detectar_color(gaussian_blur,(0,0,255), thres_color)
        resultado = aplicar_mascara(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB), mask)

        mostrar_frame("Deteccion", cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)) #Mostramos en RGB para que se vea el color correcto
        mostrar_frame("Mascara", mask)
        mostrar_frame("Resultados", resultado)

        if cv2.waitKey(1) & 0xFF == 13:
            break

    camara.stop()
    cv2.destroyAllWindows()
