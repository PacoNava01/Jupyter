'''Este  Codigo debe ser capaz de poder utilizarse empleand el modulo Hq de raspberry o una camara de uso standar como lo puede ser una webcam'''
import numpy as np
import gpiozero #Si llamamos algo es como cv2, usaremos gpiozero.lo que usemos
from time import sleep
import cv2    

'''Se comienza por importar la libreria picamera2 de raspberry, en caso de no lograrse, se dara un estado FALSE
(no se rompe el codigo, sigue corriendo sabiendo que el codigo se basara en una webcam)'''
# Intentamos importar picamera2 (solo disponible en Raspberry Pi)
try:
    from picamera2 import Picamera2
    USE_PICAMERA = True
except ImportError: #En el caso de que sea imposible importar la libreria, se genera un ImportError
    USE_PICAMERA = False
    print("Picamera2 no disponible. Se usará una webcam en su lugar\n espero tengas opencv instalado.")


# --- Inicialización de cámara ---
'''Se definen las funciones de inicialización y captura de imagenes de la camara'''
def inicializar_camara():
    if USE_PICAMERA: #Si la libreria picamera2 fue importada, if TRUE
        print("Usando Picamera2") #Mensaje para indicar la camara usada, siempre util mencionarla
        cam = Picamera2() #La camara sera dada a partir de Picamera2()
        cam.preview_configuration.main.size = (640, 480)  # Resolución reducida para eficiencia
        cam.preview_configuration.main.format = "RGB888" # Formato compatible con OpenCV
        cam.configure("preview")
        cam.start()
        return cam #Regresariamos el controlador cam
    else: #si la libreria picamera2 no fue importada, se genero un FALSE
        print("Usando webcam con OpenCV")
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cam.isOpened():
            raise Exception("No se pudo abrir la webcam.")
        return cam


def capturar_frame(camara): #Se alimenta con la variable de inicializado de la camara
    if USE_PICAMERA: #Con la picamera activada, deberemos de pasar arreglos al valor de nuestro frame
        return camara.capture_array()
    else:
        ret, frame = camara.read() #Metodo usual de lectura de camara usb
        return frame if ret else None #Si la camara usb no falla, retornamos el frame


#---conversores de colores---
def RGBaHSV(img): #Conversor de color a HSV
    return cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

def RGBaGRAY(img): #Conversor de color a GRAY
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

#--- Funciones ---
def Binarizador (frame_GRAY):#Los frames que no entren en el rango de intensidad especificado, no entrara en el threshold
    _ , thresh = cv2.threshold(frame_GRAY,60,250,cv2.THRESH_BINARY) #Probar con binarizados diferentes 
    return thresh
     
def Contornos(frame_thresh,min_area):#Debe alimentarse con una imagen binarizada
    contornos, _ = cv2.findContours(frame_thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contornos_filtrados = [c for c in contornos if cv2.contourArea(c) > min_area]
    return contornos_filtrados



def dibujar_contornos(frame, contornos): #solo dibujaremos el más grande
    if len(contornos) > 0:
        c = max(contornos, key=cv2.contourArea)  # el de mayor área directamente
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
    return frame

# def dibujar_contornos(frame, contornos): #antigua versión (no estara en uso)
#     #Reordenamos los contornos del más grande al más pequeño
#     id len(contornos) > 0:
#     c = max(contornos, key = cv2.contourArea)

#     contornos = sorted(contornos,key = lambda x:cv2.contourArea(x),reverse = True)
#     for c in contornos:
#         cv2.drawContours(frame, [c], -1, (0, 255, 0), 2) #Dibujamos el contorno sobre la imagen
#         break #Así solamente dibjaremos el contorno más grande
#     return frame    

def encajonadora (frame,contornos):
    '''Esta función recibe una lista de contornos, tomaremos el más grande 
    en cuanto a su area de ellos y a este contorno le generaremos una caja
    que acapare todos sus contornos, de modo obtengamos información sobre 
    sus parametros x,y,w,h y podamos hallar el centro de este objeto'''
    if len(contornos)>0:
        c = max(contornos,key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c) #Encontrams las coordenadas de una caja para el contorno
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) 
    return frame #Regresamos el frame con la caja ya pintada sobre él

def dibujar_centroide_mayor(frame, contornos):
    if len(contornos) > 0:
        c = max(contornos, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"({cx},{cy})", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            return frame, (cx, cy) 
            #Regresamos ademàs de la imagen regresaremos el valor de los centroides para la logica posterior
    return frame, None

#Funciòn antigua no sera usada de momento
# def dibujar_centroides(frame, contornos):
#     for c in contornos:
#         M = cv2.moments(c) #Obtenemos los "momentos" de los contornos que aparezcan en la lista de contornos
#         if M["m00"] != 0: #Si el momento obtenido es distinto de cero calculamos los centroides
#             cx = int(M["m10"] / M["m00"])
#             cy = int(M["m01"] / M["m00"])
#             cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Punto rojo
#             cv2.putText(frame, f"({cx},{cy})", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
#         break#(asì solo dibujaremos el centroide màs grande)
#     return frame

def color_mask(frame_HSV, lower, upper):
    """
    Aplica una máscara de color a una imagen en HSV usando los límites especificados.

    Parámetros:
    H:[0-179] (lo que sería 0-360/2)
    S:[0-255]
    V:[0-255]
    - lower: tupla o array con valores mínimos (H, S, V)
    - upper: tupla o array con valores máximos (H, S, V)

    Devuelve:
    - Imagen enmascarada en HSV
    """
    #Se genera la mascara que solo ocupe los colores indicados
    mask = cv2.inRange(frame_HSV, lower, upper) 
    #Regresa una imagen HSV enmascarada
    return mask


#Función que ayuda a pasar entre colores reportadas en paginas a valores aceptados por opencv
#No funciona aún
def hsv_2_opencv(lower, upper, h_max=360, s_max=100, v_max=100):
    """
    Convierte rangos HSV desde distintas escalas (como las reportadas en páginas web)
    a la escala usada por OpenCV (H: 0-179, S: 0-255, V: 0-255).

    Parámetros
    ----------
    lower : tuple/list de 3 elementos
        Límite inferior del color (H, S, V) en la escala original.
    upper : tuple/list de 3 elementos
        Límite superior del color (H, S, V) en la escala original.
    h_max : int, opcional
        Máximo valor de H en la escala de entrada (por defecto 360).
    s_max : int o float, opcional
        Máximo valor de S en la escala de entrada (por defecto 100).
    v_max : int o float, opcional
        Máximo valor de V en la escala de entrada (por defecto 100).

    Retorna
    -------
    lower_cv : tuple
        Límite inferior en la escala OpenCV.
    upper_cv : tuple
        Límite superior en la escala OpenCV.

    Ejemplo
    -------
    >>> hsv_2_opencv((0, 65, 100), (10, 100, 100))
    ((0, 166, 255), (5, 255, 255))
    """
    
    def scale_value(val, max_val, target_max):
        val = np.clip(val, 0, max_val)
        return int(round(val * target_max / max_val))

    if not (len(lower) == len(upper) == 3):
        raise ValueError("Los parámetros 'lower' y 'upper' deben tener 3 elementos cada uno (H, S, V).")

    h1, s1, v1 = lower
    h2, s2, v2 = upper

    lower_cv = (
        scale_value(h1, h_max, 179),
        scale_value(s1, s_max, 255),
        scale_value(v1, v_max, 255)
    )

    upper_cv = (
        scale_value(h2, h_max, 179),
        scale_value(s2, s_max, 255),
        scale_value(v2, v_max, 255)
    )

    return lower_cv, upper_cv

def ventanas(nombre, frame):
    '''Función para mostrar una imagen en una ventana redimensionable'''

    cv2.namedWindow(nombre, cv2.WINDOW_NORMAL)  # Crear ventana redimensionable con el nombre dado
    cv2.imshow(nombre, frame)                    # Mostrar la imagen en la ventana

#---Colores de prueba --- (rojo) es al tanteo, opencv en hsv considera rojo = verde por alguna razòn
lower_web = (100, 65, 30)
upper_web = (340, 100, 54)

lower_cv , upper_cv = hsv_2_opencv(lower_web,upper_web)
print(lower_cv , upper_cv)


#----Colores testeados----
#Rojo
low_red = np.array([50,155,84])
upper_red = np.array([179,255,255])


#--- Función del procesamiento  de la imagen ---
def procesamiento(frame,lower, upper, area_min = 400):
    '''
    Funcion encargada de ejecutar todas las funciones relacionadas
    al procesamiento de imagenes

    area_min: El area en pixeles minima para a cual se generan los contornos en el procesamiento
    lower y upper son los valores bajos y altos por lo cuales se hara el rangos de color en el procesamiento
    '''
    copyframe = frame.copy() #Copia para mostrar
    hsv_frame = RGBaHSV(frame)
    red_mask = color_mask(hsv_frame, lower, upper)
    #Creada la mascara trataremos de encontrar su centroide
    '''
    Aprovechando que ya tenemos una imagen binarizada por la mascara
    encontraremos cos ellos los contornos
    '''
    contornos_aprobados = Contornos(red_mask,400) #El segundo parametro indica el area minima en pixeles
    #Encontrados los contornos deseados ahora podemos dibujar la caja que los contienen
    copyframe = encajonadora(copyframe, contornos_aprobados)
    copyframe, centroide = dibujar_centroide_mayor(copyframe, contornos_aprobados)

    return copyframe, centroide #Regresamos los centroides y la imagen procesada

# --- Loop principal ---
try: #No esperamos un error en especifico
    cam = inicializar_camara() #Se crea una variable que se encargara de cargar el controlador de la camara, sea de raspberry o usb
    
    while True:
        frame = capturar_frame(cam) #Este sera el frame original
        if frame is None:
            print("Error: no se pudo capturar imagen.")
            break

        copyframe, centroide = procesamiento(frame,lower=low_red,upper=upper_red)

        '''
        conocemos las  dimensiones de la imagen, es más, nosotros las establecimos (640,480)
        pero si no lo hicieramos encontraremos estas dimensiones y con ellas encontrareos
        el centro de la imagen
        '''
        x, y  = frame.shape()
        center_x, center_y = x//2 , y//2

        

        ventanas("color",copyframe)

    

        if cv2.waitKey(1) == 13:  # Tecla Enter
            print("Programa cerrado, adiós")
            break

finally: #Finalmente, al terminarl el ciclo while anterior se ejecutara este bloque de cerrado de camaras

    # --- Finalización ---
    if USE_PICAMERA: #Si picamera era TRUE cerramos mediante el metodo stop()
        cam.stop()
    else: #Si se esta usando camara web, liberamos la camara con el metodod release()
        cam.release()
    cv2.destroyAllWindows() #Matamos todas las ventanas emergentes
