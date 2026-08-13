import cv2
import numpy as np
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams)

# Rutas
hef_path = "Tesis-Proyecto/Yolo_dir/yolov8n.hef"
image_path = "Tesis-Proyecto/Yolo_dir/Test.jpg"

# 1. Cargar la imagen del disco
img_original = cv2.imread(image_path)
if img_original is None:
    raise FileNotFoundError(f"No se pudo encontrar la imagen en: {image_path}")

# OpenCV lee en BGR por defecto. Convertimos a RGB porque la mayoría de redes NPU se entrenan en RGB
img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

# 2. Configurar Hailo NPU
hef = HEF(hef_path)
input_info = hef.get_input_vstream_infos()[0]
input_h, input_w = input_info.shape[0], input_info.shape[1]
input_name = input_info.name

# 3. Pre-procesamiento
# Redimensionamos la imagen a la resolución que exige la NPU (ej. 640x640)
resized_img = cv2.resize(img_rgb, (input_w, input_h))
# Añadimos la dimensión de lote (Batch): de (640,640,3) a (1,640,640,3)
input_data = {input_name: np.expand_dims(resized_img, axis=0).astype(np.uint8)}

# 4. Inferencia en el Hardware (PCIe)
params = VDevice.create_params()
with VDevice(params) as target:
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_vstream_params = InputVStreamParams.make(network_group)
    output_vstream_params = OutputVStreamParams.make(network_group)

    with network_group.activate(network_group_params):
        with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
            print("Enviando 'test.jpg' a la NPU Hailo-8L...")
            raw_results = infer_pipeline.infer(input_data)
            print("¡Inferencia completada con éxito!")

# 5. Analizar el resultado puro
for key, value in raw_results.items():
    print(f"\nClave de salida: '{key}'")
    print(f"Tipo de objeto devuelto: {type(value)}")
    if isinstance(value, np.ndarray):
        print(f"Shape del arreglo NumPy: {value.shape}")
        print(f"Ejemplo de primer elemento: {value[0][:2]}")
    elif isinstance(value, list):
        print(f"Longitud de la lista devuelta: {len(value)}")
        print(f"Tipo del primer elemento dentro de la lista: {type(value[0])}")

'''
(.pacon) pacon@Pacon:~/Jupyter $ /home/pacon/Jupyter/Librerias/.pacon/bin/python /home/pacon/Jupyter/Tesis-Proyecto/Yolo_dir/Yolo_test.py
Enviando 'test.jpg' a la NPU Hailo-8L...
¡Inferencia completada con éxito!

Clave de salida: 'yolov8n/yolov8_nms_postprocess'
Tipo de objeto devuelto: <class 'list'>
Longitud de la lista devuelta: 1
Tipo del primer elemento dentro de la lista: <class 'list'>
'''