import cv2
import numpy as np
from hailo_platform import (HEF,VDevice,HailoStreamInterface,InferVStreams,
                            ConfigureParams,InputVStreamParams,OutputVStreamParams)

#Rutas
hef_path = "Tesis-Proyecto/Yolo_dir/yolov8n.hef"
image_path = "Tesis-Proyecto/Yolo_dir/Test.jpg"

#Cargar imagen del disco
img_original = cv2.imread(image_path)
if img_original is None:
    raise FileNotFoundError(f"No se pudo encontrar la imagen en {image_path}")

#Opencv lee en BGR por defecto. Convertimos en RGB porque la mayoria de redes NPU se entrenan con RGB
img_rgb = cv2.cvtColor(img_original,cv2.COLOR_BAYER_BG2BGR)

#Configurar Hailo NPU
hef = HEF(hef_path)
input_info = hef.get_input_stream_infos()[0]
input_h,input_w = input_info.shape[0],input_info.shape[1]
input_name = input_info.name

#Preprocesamiento 
#redimensionar la imagen a la resolucion que exge la NPU
resized_img = cv2.resize(img_rgb,(input_w,input_h))
#Añadimos la dimension del lote (batch) de (640,640,3) a (1,640,640,3)
input_data = {input_name: np.expand_dims(resized_img,axis=0).astype(np.uint8)}

#Inferencia en el hardware (pcie)
params = VDevice.create_params()
with VDevice(params) as target:
    configure_params = ConfigureParams.create_from_hef(hef,interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef,configure_params)[0]
    network_group_params = network_group.create_params()

    input_vstream_params = InputVStreamParams.make(network_group)
    output_vstream_params = OutputVStreamParams.make(network_group)

    with network_group.activate(network_group_params):
        with InferVStreams(network_group,input_vstream_params,output_vstream_params) as infer_pipeline:
            print("Envianto 'Test.jpg' a la NPU de HAILO")
            raw_results = infer_pipeline.infer(input_data)
            print("Inferencia completaa con exito")
#Analizar el resultado puro
for key, value in raw_results.items():
    print(f"\nClave de salida: '{key}'")
    print(f"Tipo de objeto devuelto: {type(value)}")
    if isinstance(value,np.ndarray):
        print(f"Shaoe del arreglo Numpy: {value.shape}")
        print(f"Ejemplo de primer elemento: {value[0][:2]}")
    elif isinstance(value,list):
        print(f"Longitud de la lista devuelta: {len(value)}")
        print(f"Tipo del primer elemento dentro de la lista: {type(value[0])}")

