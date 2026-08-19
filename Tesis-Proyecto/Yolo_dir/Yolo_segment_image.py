import cv2
import numpy as np
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams,
    ConfigureParams, InputVStreamParams, OutputVStreamParams
)

# 80 Clases estándar de COCO
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

# Rutas de entrada y salida
HEF_PATH = "/home/pacon/Jupyter/Tesis-Proyecto/Yolo_dir/yolov8n_seg.hef"
IMAGE_PATH = "/home/pacon/Jupyter/Tesis-Proyecto/Yolo_dir/test_game.jpg"
OUTPUT_PATH = "/home/pacon/Jupyter/Tesis-Proyecto/Yolo_dir/test_segmented.jpg"

CONF_THRESHOLD = 0.35
MASK_THRESHOLD = 0.50

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def run_segmentation():
    # 1. Cargar la imagen original
    img_original = cv2.imread(IMAGE_PATH)
    if img_original is None:
        raise FileNotFoundError(f"No se encontró la imagen en: {IMAGE_PATH}")
    
    h_orig, w_orig, _ = img_original.shape
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

    # 2. Cargar e inspeccionar el modelo HEF
    hef = HEF(HEF_PATH)
    input_info = hef.get_input_vstream_infos()[0]
    input_h, input_w = input_info.shape[0], input_info.shape[1]
    input_name = input_info.name

    # Pre-procesamiento: redimensionar y empaquetar en Batch de 1
    resized_img = cv2.resize(img_rgb, (input_w, input_h))
    input_data = {input_name: np.expand_dims(resized_img, axis=0).astype(np.uint8)}

    # 3. Configurar e inicializar la NPU Hailo-8L vía PCIe
    params = VDevice.create_params()
    with VDevice(params) as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstream_params = InputVStreamParams.make(network_group)
        output_vstream_params = OutputVStreamParams.make(network_group)

        with network_group.activate(network_group_params):
            with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
                print("Ejecutando inferencia de segmentación en la NPU...")
                raw_results = infer_pipeline.infer(input_data)
                print("Inferencia completada.")

    # 4. Procesamiento de salidas
    output_img = img_original.copy()
    overlay = img_original.copy()

    # Identificar tensores de salida (Proto masks vs Detecciones NMS)
    proto_mask = None
    detections = None

    for key, value in raw_results.items():
        tensor = np.array(value)
        if 32 in tensor.shape and len(tensor.shape) >= 3:
            proto_mask = tensor[0] if tensor.ndim == 4 else tensor
        else:
            detections = value[0] if isinstance(value, list) else tensor[0]

    # Procesar detecciones y dibujar máscaras
    if detections is not None:
        det_array = np.array(detections)
        
        if det_array.ndim >= 2:
            for class_id in range(det_array.shape[0]):
                boxes = det_array[class_id]
                for det in boxes:
                    if len(det) >= 5:
                        ymin, xmin, ymax, xmax, score = det[:5]
                        
                        if score >= CONF_THRESHOLD:
                            # Normalizar de forma segura las coordenadas (soporta formato 0-1 o escala de red)
                            if xmin <= 1.0 and xmax <= 1.0:
                                n_xmin, n_ymin, n_xmax, n_ymax = xmin, ymin, xmax, ymax
                            else:
                                n_xmin = xmin / float(input_w)
                                n_ymin = ymin / float(input_h)
                                n_xmax = xmax / float(input_w)
                                n_ymax = ymax / float(input_h)

                            # Coordenadas escaladas a la imagen original
                            x1 = max(0, int(n_xmin * w_orig))
                            y1 = max(0, int(n_ymin * h_orig))
                            x2 = min(w_orig, int(n_xmax * w_orig))
                            y2 = min(h_orig, int(n_ymax * h_orig))

                            # Color aleatorio pseudo-único por clase
                            np.random.seed(int(class_id))
                            color = [int(c) for c in np.random.randint(50, 255, size=3)]

                            # Dibujar caja delimitadora
                            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)

                            # Si hay tensor de prototipos disponible, generar la máscara poligonal
                            if proto_mask is not None:
                                mask_h, mask_w, num_proto = proto_mask.shape
                                mx1 = max(0, min(int(n_xmin * mask_w), mask_w))
                                my1 = max(0, min(int(n_ymin * mask_h), mask_h))
                                mx2 = max(0, min(int(n_xmax * mask_w), mask_w))
                                my2 = max(0, min(int(n_ymax * mask_h), mask_h))

                                if mx2 > mx1 and my2 > my1:
                                    sub_proto = proto_mask[my1:my2, mx1:mx2, :]
                                    mask_crop = np.mean(sub_proto, axis=-1)
                                    mask_binary = (sigmoid(mask_crop) > MASK_THRESHOLD).astype(np.uint8)
                                    
                                    # Asegurar dimensiones válidas para el redimensionamiento
                                    box_w = max(1, x2 - x1)
                                    box_h = max(1, y2 - y1)
                                    mask_resized = cv2.resize(mask_binary, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
                                    
                                    roi = overlay[y1:y2, x1:x2]
                                    if roi.shape[0] == mask_resized.shape[0] and roi.shape[1] == mask_resized.shape[1]:
                                        roi[mask_resized == 1] = color
                            
                            # Etiqueta de texto
                            label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"ID:{class_id}"
                            caption = f"{label} {score:.2f}"
                            cv2.putText(output_img, caption, (x1, max(y1 - 6, 15)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Fusionar la capa coloreada de máscaras con transparencia (alpha blending)
        alpha = 0.45
        cv2.addWeighted(overlay, alpha, output_img, 1 - alpha, 0, output_img)

    # 5. Guardar la imagen final
    cv2.imwrite(OUTPUT_PATH, output_img)
    print(f"Segmentación finalizada con éxito. Imagen guardada en: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_segmentation()