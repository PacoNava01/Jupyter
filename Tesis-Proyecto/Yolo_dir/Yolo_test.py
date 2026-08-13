import cv2
import numpy as np
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams)

# COCO Classes (Index 2 es 'car', Index 7 es 'truck', Index 5 es 'bus')
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

hef_path = "Tesis-Proyecto/Yolo_dir/yolov8n.hef"
image_path = "Tesis-Proyecto/Yolo_dir/test.jpg"

img_original = cv2.imread(image_path)
h_orig, w_orig, _ = img_original.shape
img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

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

# Copia de la imagen original en BGR para escribir los resultados
output_img = img_original.copy()
CONF_THRESHOLD = 0.35  # Umbral para filtrar falsos positivos

# Recorrer los resultados
for key, detections in raw_results.items():
    # Desempaquetar lote
    for batch in detections:
        for det in batch:
            # Extraer valores numéricos
            try:
                if hasattr(det, 'get_bbox'):
                    bbox = det.get_bbox()
                    ymin, xmin, ymax, xmax = bbox.ymin(), bbox.xmin(), bbox.ymax(), bbox.xmax()
                    score = det.get_confidence()
                    class_id = int(det.get_class_id())
                else:
                    ymin, xmin, ymax, xmax, score, class_id = det[:6]
                    class_id = int(class_id)
            except Exception:
                continue

            if score >= CONF_THRESHOLD:
                # Escalar coordenadas [0.0 - 1.0] a píxeles reales de test.jpg
                x1 = int(xmin * w_orig)
                y1 = int(ymin * h_orig)
                x2 = int(xmax * w_orig)
                y2 = int(ymax * h_orig)

                label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"ID:{class_id}"
                caption = f"{label} {score:.2f}"

                # Dibujar recuadro y texto en azul/verde
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_img, caption, (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Guardar imagen resultante
save_path = "Tesis-Proyecto/Yolo_dir/test_detected.jpg"
cv2.imwrite(save_path, output_img)
print(f"Detección finalizada. Revisa la imagen guardada en: {save_path}")
