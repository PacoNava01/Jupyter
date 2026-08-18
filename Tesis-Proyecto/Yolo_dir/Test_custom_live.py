import os
import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams,
    ConfigureParams, InputVStreamParams, OutputVStreamParams
)

# 1. Configuración de Clases y Colores para Color Cap (11 clases)
CUSTOM_CLASSES = [
    'black', 'blue', 'brown', 'green', 'grey', 
    'orange', 'pink', 'purple', 'red', 'white', 'yellow'
]

# Mapa de colores BGR para visualización en OpenCV
CLASS_COLORS = {
    0: (30, 30, 30),       # black
    1: (255, 0, 0),        # blue
    2: (19, 69, 139),      # brown
    3: (0, 255, 0),        # green
    4: (128, 128, 128),    # grey
    5: (0, 165, 255),      # orange
    6: (203, 192, 255),    # pink
    7: (128, 0, 128),      # purple
    8: (0, 0, 255),        # red
    9: (255, 255, 255),    # white
    10: (0, 255, 255)      # yellow
}

NUM_CLASSES = len(CUSTOM_CLASSES)
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45

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

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_grid_strides(img_size=640, strides=(8, 16, 32)):
    grid_points, stride_tensor = [], []
    for s in strides:
        grid_dim = img_size // s
        y, x = np.meshgrid(np.arange(grid_dim), np.arange(grid_dim), indexing='ij')
        grid_points.append(np.stack([x.flatten(), y.flatten()], axis=-1))
        stride_tensor.append(np.full((grid_dim * grid_dim, 1), s))
    return np.vstack(grid_points).astype(np.float32), np.vstack(stride_tensor).astype(np.float32)

# 2. Configuración del modelo HEF
hef_path = "Tesis-Proyecto/Yolo_dir/best_color_cap.hef"
if not os.path.exists(hef_path):
    # Fallback a ruta relativa simple si se ejecuta directamente dentro de Yolo_dir
    hef_path = "best_color_cap.hef"

hef = HEF(hef_path)
input_info = hef.get_input_vstream_infos()[0]
input_h, input_w = input_info.shape[0], input_info.shape[1]
input_name = input_info.name

# Generar cuadrículas de anclas y pesos DFL precomputados
anchors, strides = generate_grid_strides(img_size=input_w)
dfl_weights = np.arange(16, dtype=np.float32)

# 3. Inicialización del Hardware Hailo-8L
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

            print("Iniciando detección Color Cap en vivo (Presiona ENTER para salir)...")

            try:
                while True:
                    # Captura en RGB nativo desde Picamera2
                    frame_rgb = cam.capture_array()
                    if frame_rgb is None:
                        break

                    # Orientación física de la cámara
                    frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
                    h_orig, w_orig, _ = frame_rgb.shape

                    # Preprocesamiento NPU (uint8 RGB)
                    resized_img = cv2.resize(frame_rgb, (input_w, input_h))
                    input_data = {input_name: np.expand_dims(resized_img, axis=0).astype(np.uint8)}
                    raw_results = infer_pipeline.infer(input_data)

                    # Separar tensores por número de canales
                    cls_outputs_raw = []
                    box_outputs_raw = []

                    for name, tensor in raw_results.items():
                        t = tensor[0]
                        if t.shape[-1] == NUM_CLASSES:
                            cls_outputs_raw.append(t)
                        elif t.shape[-1] in (16, 64):
                            box_outputs_raw.append(t)

                    # Ordenar descendentemente por resolución espacial (80x80 -> 40x40 -> 20x20)
                    cls_outputs_raw.sort(key=lambda x: x.shape[0], reverse=True)
                    box_outputs_raw.sort(key=lambda x: x.shape[0], reverse=True)

                    if cls_outputs_raw and box_outputs_raw:
                        # Vectorizar clases
                        cls_concat = np.vstack([t.reshape(-1, NUM_CLASSES) for t in cls_outputs_raw])
                        all_cls = sigmoid(cls_concat)

                        # Vectorizar y decodificar DFL
                        box_concat = np.vstack([t.reshape(-1, 64) for t in box_outputs_raw])
                        dfl_reshaped = box_concat.reshape(-1, 4, 16)
                        exp_dfl = np.exp(dfl_reshaped - np.max(dfl_reshaped, axis=-1, keepdims=True))
                        dfl_softmax = exp_dfl / np.sum(exp_dfl, axis=-1, keepdims=True)
                        dist = np.sum(dfl_softmax * dfl_weights, axis=-1)

                        # Decodificación de coordenadas relativas a absolutas escaladas
                        x1 = (anchors[:, 0] - dist[:, 0]) * strides[:, 0] * (float(w_orig) / input_w)
                        y1 = (anchors[:, 1] - dist[:, 1]) * strides[:, 0] * (float(h_orig) / input_h)
                        x2 = (anchors[:, 0] + dist[:, 2]) * strides[:, 0] * (float(w_orig) / input_w)
                        y2 = (anchors[:, 1] + dist[:, 3]) * strides[:, 0] * (float(h_orig) / input_h)

                        boxes = np.column_stack([x1, y1, x2 - x1, y2 - y1]).astype(int)

                        # Preparar canvas BGR solo para visualización en OpenCV
                        display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                        # Supresión de No Máximos (NMS) por cada clase
                        for class_id in range(NUM_CLASSES):
                            scores = all_cls[:, class_id]
                            mask = scores >= CONF_THRESHOLD
                            
                            if np.any(mask):
                                filtered_boxes = boxes[mask].tolist()
                                filtered_scores = scores[mask].astype(float).tolist()

                                indices = cv2.dnn.NMSBoxes(
                                    filtered_boxes, filtered_scores, CONF_THRESHOLD, IOU_THRESHOLD
                                )

                                for idx in indices:
                                    i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
                                    bx, by, bw, bh = filtered_boxes[i]
                                    score = filtered_scores[i]
                                    label_name = CUSTOM_CLASSES[class_id]
                                    color = CLASS_COLORS.get(class_id, (0, 255, 0))

                                    # Dibujar caja y etiqueta
                                    cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), color, 2)
                                    caption = f"{label_name} {score:.2f}"
                                    cv2.putText(
                                        display_frame, caption,
                                        (bx, max(by - 8, 15)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                                    )

                        cv2.imshow("Hailo-8L - Deteccion Color Cap", display_frame)
                    else:
                        
                        cv2.imshow("Hailo-8L - Deteccion Color Cap", display_frame)

                    if cv2.waitKey(1) & 0xFF == 13:  # Enter para salir
                        break
            finally:
                cam.stop()
                cv2.destroyAllWindows()
                print("Cámara liberada y recursos cerrados.")