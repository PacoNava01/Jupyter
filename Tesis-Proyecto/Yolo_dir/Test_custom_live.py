import cv2 
import numpy as np
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams)

#1. Definicion de las 3 clases personalizadas
CUSTOM_CLASSES = ["red","green","blue"]

#Colores BGR para dibujar cada clase: rojo (B=0,G=0,R=255), verde (B=0,G=255,R=0), azul (B=255,G=0,R=0)
CLASS_COLORS = {
    0:(0,0,255), #Rojo
    1:(0,255,0), #Verde
    2:(255,0,0) #Azull
}

def init_cam():
    try:
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        print("Cámara Picamera2 iniciada.")
        return cam
    except Exception as e:
        print(f"Error iniciando la cámara: {e}")
        return None

#2. Cargar el modelo compilado .hef
hef_path = "Tesis-Proyecto/Yolo_dir/yolov8n.hef"
hef = HEF(hef_path)

input_info = hef.get_input_vstream_infos()[0]
input_h, input_w = input_info.shape[0], input_info.shape[1]
input_name = input_info.name

#Inspeccionar nombre de capas de salida
output_infos = hef.get_output_vstream_infos()
print(f"Capas de salida detectadas en el HEF: {[info.name for info in output_infos]}")

#3. Configurar el Hardware Hailo-8L
params = VDevice.create_params()
with VDevice(params) as target:
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_vstream_params = InputVStreamParams.make(network_group)
    output_vstream_params = OutputVStreamParams.make(network_group)

    with network_group.activate(network_group_params):
        with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
            
            cam = init_cam()
            if cam is None:
                exit(1)

            CONF_THRESHOLD = 0.35 #Umbral de confianza
            print("Iniciando detección de colores RGB en vivo (Presiona ENTER para salir)...")

            try:
                while True:
                    frame = cam.capture_array() #Frame en RGB desde Picamera2
                    if frame is None:
                        break
                    h_orig,w_orig, _ = frame.shape

                    #Preprocesamiento hacia la NPU
                    resized_img = cv2.resize(frame,(input_w,input_h))
                    input_data = {input_name: np.expand_dims(resized_img,axis=0).astype(np.uint8)}

                    #Inferencia en el chip Hailo-8L
                    raw_results = infer_pipeline.infer(input_data)

                    #frame para Opencv (RGB a BGR)
                    display_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

                    #Desempaquetado de las saldias
                    for out_name, out_tensor in raw_results.item():
                        #Si el HEF postproceso NMS agrupado por clases (3 clases en este caso)
                        if isinstance