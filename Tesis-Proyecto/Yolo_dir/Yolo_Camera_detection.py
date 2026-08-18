import os
import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams)

# Clases COCO
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Cargar modelo
hef_path = os.path.join(BASE_DIR, "yolov8n.hef")
hef = HEF(hef_path)

input_info = hef.get_input_vstream_infos()[0]
input_h, input_w = input_info.shape[0], input_info.shape[1]
input_name = input_info.name
output_key = 'yolov8n/yolov8_nms_postprocess'

# 2. Configurar hardware NPU
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

            CONF_THRESHOLD = 0.4  # Umbral recomendado para tiempo real
            print("Iniciando inferencia en vivo (Presiona ENTER para salir)...")

            try:
                while True:
                    # Captura de frame en RGB
                    frame = cam.capture_array()
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                    if frame is None:
                        break

                    h_orig, w_orig, _ = frame.shape

                    # Preprocesamiento NPU
                    resized_img = cv2.resize(frame, (input_w, input_h))
                    input_data = {input_name: np.expand_dims(resized_img, axis=0).astype(np.uint8)}

                    # Inferencia en hardware
                    raw_results = infer_pipeline.infer(input_data)

                    # Preparar frame para visualización (RGB a BGR)
                    display_frame = frame

                    # Postprocesamiento seguro (evita errores con formas heterogéneas)
                    detections_batch = raw_results[output_key]
                    current_detections = detections_batch[0] if isinstance(detections_batch, list) else detections_batch

                    for class_id, boxes_for_class in enumerate(current_detections):
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

                                    label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"ID:{class_id}"
                                    caption = f"{label} {score:.2f}"

                                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(display_frame, caption, (x1, max(y1 - 5, 15)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    cv2.imshow("Hailo AI Kit - Deteccion en Vivo", display_frame)
                    
                    # Tecla Enter para finalizar
                    if cv2.waitKey(1) & 0xFF == 13:
                        break
            finally:
                cam.stop()
                cv2.destroyAllWindows()
                print("Cámara liberada correctamente.")