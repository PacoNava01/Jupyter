import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams)

# 1. Definición de tus 3 clases personalizadas
CUSTOM_CLASSES = ["red", "green", "blue"]

# Colores BGR para dibujar cada clase: Rojo (B=0,G=0,R=255), Verde (B=0,G=255,R=0), Azul (B=255,G=0,R=0)
CLASS_COLORS = {
    0: (0, 0, 255),    # Red
    1: (0, 255, 0),    # Green
    2: (255, 0, 0)     # Blue
}

def init_cam():
    try:
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        print("Cámara Picamera2 iniciada correctamente.")
        return cam
    except Exception as e:
        print(f"Error al iniciar la cámara: {e}")
        return None

# 2. Cargar tu modelo compilado .hef
hef_path = "Tesis-Proyecto/Yolo_dir/best_rgb012.hef"
hef = HEF(hef_path)

input_info = hef.get_input_vstream_infos()[0]
input_h, input_w = input_info.shape[0], input_info.shape[1]
input_name = input_info.name

# Inspeccionar nombre de capas de salida
output_infos = hef.get_output_vstream_infos()
print(f"Capas de salida detectadas en el HEF: {[info.name for info in output_infos]}")

# 3. Configurar Hardware Hailo-8L
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

            CONF_THRESHOLD = 0.2  # Umbral de confianza
            print("Iniciando detección de colores RGB en vivo (Presiona ENTER para salir)...")

            try:
                while True:
                    frame = cam.capture_array() # Frame en RGB desde Picamera2
                    if frame is None:
                        break
                        
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    h_orig, w_orig, _ = frame.shape

                    # Preprocesamiento hacia la NPU (YOLO espera RGB)
                    resized_img = cv2.resize(frame, (input_w, input_h))
                    input_data = {input_name: np.expand_dims(resized_img, axis=0).astype(np.uint8)}

                    # Inferencia en el chip Hailo-8L
                    raw_results = infer_pipeline.infer(input_data)

                    # CORREGIDO: Convertir de RGB (Picamera2) a BGR (para que OpenCV pinte bien los colores)
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Desempaquetado de las salidas
                    for out_name, out_tensor in raw_results.items():
                        if isinstance(out_tensor, (list, np.ndarray)) and len(out_tensor) > 0:
                            detections_batch = out_tensor[0]

                            # Recorrer las clases detectadas de forma segura
                            for class_id, boxes_for_class in enumerate(detections_batch):
                                if class_id >= len(CUSTOM_CLASSES):
                                    break
                                
                                if len(boxes_for_class) == 0:
                                    continue
                                    
                                for det in boxes_for_class:
                                    if len(det) >= 5:
                                        ymin, xmin, ymax, xmax, score = det[:5]

                                        if score >= CONF_THRESHOLD:
                                            x1 = int(xmin * w_orig)
                                            y1 = int(ymin * h_orig)
                                            x2 = int(xmax * w_orig)
                                            y2 = int(ymax * h_orig)

                                            label_name = CUSTOM_CLASSES[class_id]
                                            color_box = CLASS_COLORS.get(class_id, (0, 255, 0))
                                            caption = f"{label_name} {score:.2f}"

                                            # Dibujar cuadro delimitador y etiqueta
                                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color_box, 2)
                                            cv2.putText(display_frame, caption, (x1, max(y1 - 8, 15)),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

                    cv2.imshow("Hailo AI Kit - Deteccion RGB012", display_frame)

                    # Salir con la tecla Enter (13)
                    if cv2.waitKey(1) & 0xFF == 13:
                        break
            finally:
                cam.stop()
                cv2.destroyAllWindows()
                print("Cámara liberada y recursos cerrados.")

                '''
                (.pacon) pacon@Pacon:~/Jupyter $ /home/pacon/Jupyter/Librerias/.pacon/bin/python /home/pacon/Jupyter/Tesis-Proyecto/Yolo_dir/Test_custom_live.py
Capas de salida detectadas en el HEF: ['yolov8s/conv41', 'yolov8s/conv42', 'yolov8s/conv52', 'yolov8s/conv53', 'yolov8s/conv62', 'yolov8s/conv63']
[3:15:34.693455167] [63500]  INFO Camera camera_manager.cpp:340 libcamera v0.7.1+rpt20260609
[3:15:34.702739199] [63518]  INFO RPI pisp.cpp:720 libpisp version v1.6.0 29-06-2026 (16:17:40)
[3:15:34.711715212] [63518]  INFO IPAProxy ipa_proxy.cpp:184 Using tuning file /usr/share/libcamera/ipa/rpi/pisp/imx477.json
[3:15:34.720775022] [63518]  INFO Camera camera_manager.cpp:223 Adding camera '/base/axi/pcie@1000120000/rp1/i2c@88000/imx477@1a' for pipeline handler rpi/pisp
[3:15:34.720832615] [63518]  INFO RPI pisp.cpp:1181 Registered camera /base/axi/pcie@1000120000/rp1/i2c@88000/imx477@1a to CFE device /dev/media0 and ISP device /dev/media2 using PiSP variant BCM2712_C0
[3:15:34.724297823] [63518]  WARN V4L2 v4l2_pixelformat.cpp:346 Unsupported V4L2 pixel format Nc30
[3:15:34.724342583] [63518]  WARN V4L2 v4l2_pixelformat.cpp:346 Unsupported V4L2 pixel format Nc12
[3:15:34.726709457] [63500]  INFO Camera camera.cpp:1216 configuring streams: (0) 640x480-RGB888/SMPTE170M/Rec709/None/Full (1) 1332x990-BGGR_PISP_COMP1/RAW
[3:15:34.728090348] [63518]  INFO RPI pisp.cpp:1485 Sensor: /base/axi/pcie@1000120000/rp1/i2c@88000/imx477@1a - Selected sensor format: 1332x990-SBGGR12_1X12/RAW - Selected CFE format: 1332x990-PC1B/RAW
Cámara Picamera2 iniciada correctamente.
Iniciando detección de colores RGB en vivo (Presiona ENTER para salir)...
Cámara liberada y recursos cerrados.
Traceback (most recent call last):
  File "/home/pacon/Jupyter/Tesis-Proyecto/Yolo_dir/Test_custom_live.py", line 100, in <module>
    x1 = int(xmin * w_orig)
             ~~~~~^~~~~~~~
OverflowError: Python integer 640 out of bounds for uint8
                '''