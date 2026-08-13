import cv2
import numpy as np
from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams

input_info = hef.get_input_stream_infos()[0]
input_height, input_width = input_info.shape[0], input_info.shape[1]
input_name = input_info.name

input_vstream_params = InputVStreamParams.make(network_group)
output_vstream_params = OutputVStreamParams.make(network_group)

with network_group.activate(network_group.create_params()):
    with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
        # Cargar imagen de prueba
        img = cv2.imread("test.jpg")  # pon una foto con objetos conocidos
        resized = cv2.resize(img, (input_width, input_height))
        normalized = resized.astype(np.float32) / 255.0
        input_data = {input_name: np.expand_dims(normalized, axis=0)}

        raw_results = infer_pipeline.infer(input_data)
        print("Claves de salida:", raw_results.keys())
