import cv2
import numpy as np
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams)

# Clases del dataset COCO (80 clases estándar)
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

hef_path = "/home/pacon/Jupyter/Tesis-Proyecto/Yolo_dir/yolov8n.hef"
image_path = "/home/pacon/Jupyter/Tesis-Proyecto/Yolo_dir/test_game.jpg"
output_path = "Tesis-Proyecto/Yolo_dir/test_game_detected.jpg"

# 1. Cargar imagen original
img_original = cv2.imread(image_path)
if img_original is None:
    raise FileNotFoundError(f"No se encontró la imagen en {image_path}")

h_orig, w_orig, _ = img_original.shape
img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

# 2. Cargar modelo e inferir en la NPU
hef = HEF(hef_path)
input_info = hef.get_input_vstream_infos()[0]
input_h, input_w = input_info.shape[0], input_info.shape[1]
input_name = input_info.name

resized_img = cv2.resize(img_rgb, (input_w, input_h))
input_data = {input_name: np.expand_dims(resized_img, axis=0).astype(np.uint8)}

params = VDevice.create_params()
with VDevice(params) as target:
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_vstream_params = InputVStreamParams.make(network_group)
    output_vstream_params = OutputVStreamParams.make(network_group)

    with network_group.activate(network_group_params):
        with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
            raw_results = infer_pipeline.infer(input_data)

# 3. Post-procesamiento e inspección de cajas
CONF_THRESHOLD = 0.3  # Umbral de confianza
output_img = img_original.copy()
output_key = 'yolov8n/yolov8_nms_postprocess'

# Extraemos el primer elemento del batch
detections_batch = raw_results[output_key][0]
total_detecciones = 0

# Iteramos directamente sobre la lista de clases sin forzar un numpy array homogéneo
for class_id, boxes_for_class in enumerate(detections_batch):
    
    for det in boxes_for_class:
        # Extraer [ymin, xmin, ymax, xmax, score]
        if len(det) >= 5:
            ymin, xmin, ymax, xmax, score = det[:5]
            
            if score >= CONF_THRESHOLD:
                total_detecciones += 1
                
                # Desnormalizar coordenadas (de 0.0 - 1.0 a píxeles de test.jpg)
                x1 = int(xmin * w_orig)
                y1 = int(ymin * h_orig)
                x2 = int(xmax * w_orig)
                y2 = int(ymax * h_orig)

                label_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"ID:{class_id}"
                caption = f"{label_name} {score:.2f}"

                # Dibujar rectángulo verde y texto
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_img, caption, (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print(f"Total de objetos detectados con éxito: {total_detecciones}")

# 4. Guardar la imagen con las marcas
cv2.imwrite(output_path, output_img)
print(f"Imagen procesada guardada en: {output_path}")