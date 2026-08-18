import os
import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams,
    ConfigureParams, InputVStreamParams, OutputVStreamParams
)

# 1. Clases de Color Cap (11 Clases)
CUSTOM_CLASSES = [
    'black', 'blue', 'brown', 'green', 'grey', 
    'orange', 'pink', 'purple', 'red', 'white', 'yellow'
]

# Mapa de colores BGR (OpenCV) para dibujar las cajas
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
CONF_THRESHOLD = 0.30  # Umbral inicial más permisivo para pruebas
IOU_THRESHOLD = 0.45

def init_cam():
    try:
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        print("--> Cámara Picamera2 inicializada correctamente.")
        return cam
    except Exception as e:
        print(f"Error al iniciar Picamera2: {e}")
        return None

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

def generate_grid_strides(img_size=640, strides=(8, 16, 32)):
    grid_points, stride_tensor = [], []
    for s in strides:
        grid_dim = img_size // s
        # Indexing 'xy' genera (col, fila) -> (x, y)
        x, y = np.meshgrid(np.arange(grid_dim), np.arange(grid_dim), indexing='xy')
        grid_points.append(np.stack([x.flatten(), y.flatten()], axis=-1))
        stride_tensor.append(np.full((grid_dim * grid_dim, 1), s))
    return np.vstack(grid_points).astype(np.float32), np.vstack(stride_tensor).astype(np.float32)

# 2. Cargar modelo HEF
hef_path = "best_color_cap.hef"
if not os.path.exists(hef_path):
    hef_path = "Tesis-Proyecto/Yolo_dir/best_color_cap.hef"

if not os.path.exists(hef_path):
    raise FileNotFoundError(f"No se encontró el archivo HEF en la ruta especificada.")

hef = HEF(hef_path)
input_info = hef.get_input_vstream_infos()[0]
input_h, input_w = input_info.shape[0], input_info.shape[1]
input_name = input_info.name

anchors, strides = generate_grid_strides(img_size=input_w)
dfl_weights = np.arange(16, dtype=np.float32)

# 3. Inicializar VDevice e inferencia
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

            print("\n--> INFERENCIA ACTIVA. Mostrando video natural en OpenCV...")
            frame_count = 0

            try:
                while True:
                    # 1. Capturar RGB nativo de Picamera2
                    frame_rgb = cam.capture_array()
                    if frame_rgb is None:
                        break

                    # Orientar física de montaje si está invertida
                    frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
                    h_orig, w_orig, _ = frame_rgb.shape

                    # 2. Canvas de visualización en BGR natural
                    display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    # 3. Preprocesamiento para la NPU (RGB estricto)
                    resized_rgb = cv2.resize(frame_rgb, (input_w, input_h))
                    input_data = {input_name: np.expand_dims(resized_rgb, axis=0).astype(np.uint8)}
                    raw_results = infer_pipeline.infer(input_data)

                    # 4. Clasificar tensores por resolución espacial (80, 40, 20)
                    scales = [80, 40, 20]
                    cls_dict = {}
                    box_dict = {}

                    for name, tensor in raw_results.items():
                        t = tensor[0]
                        dim = t.shape[0]
                        if t.shape[-1] == NUM_CLASSES:
                            cls_dict[dim] = t
                        elif t.shape[-1] in (16, 64):
                            box_dict[dim] = t

                    # Verificar que todas las escalas estén presentes
                    if all(s in cls_dict for s in scales) and all(s in box_dict for s in scales):
                        cls_outputs = [cls_dict[s].reshape(-1, NUM_CLASSES) for s in scales]
                        box_outputs = [box_dict[s].reshape(-1, 64) for s in scales]

                        all_cls = sigmoid(np.vstack(cls_outputs))
                        all_boxes_raw = np.vstack(box_outputs)

                        # Decodificación DFL
                        dfl_reshaped = all_boxes_raw.reshape(-1, 4, 16)
                        exp_dfl = np.exp(dfl_reshaped - np.max(dfl_reshaped, axis=-1, keepdims=True))
                        dfl_softmax = exp_dfl / np.sum(exp_dfl, axis=-1, keepdims=True)
                        dist = np.sum(dfl_softmax * dfl_weights, axis=-1)

                        # Coordenadas relativas a absolutas en la resolución de cámara
                        x1 = (anchors[:, 0] - dist[:, 0]) * strides[:, 0] * (float(w_orig) / input_w)
                        y1 = (anchors[:, 1] - dist[:, 1]) * strides[:, 0] * (float(h_orig) / input_h)
                        x2 = (anchors[:, 0] + dist[:, 2]) * strides[:, 0] * (float(w_orig) / input_w)
                        y2 = (anchors[:, 1] + dist[:, 3]) * strides[:, 0] * (float(h_orig) / input_h)

                        boxes = np.column_stack([x1, y1, x2 - x1, y2 - y1]).astype(int)

                        # Diagnóstico en consola cada 30 cuadros
                        frame_count += 1
                        if frame_count % 30 == 0:
                            max_conf = np.max(all_cls)
                            pred_cls = np.argmax(np.max(all_cls, axis=0))
                            print(f"[Diag] Max Confianza: {max_conf:.3f} | Clase estimada: {CUSTOM_CLASSES[pred_cls]}")

                        # NMS y dibujo de cajas
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

                                    cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), color, 2)
                                    caption = f"{label_name} {score:.2f}"
                                    cv2.putText(
                                        display_frame, caption,
                                        (bx, max(by - 8, 15)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                                    )

                    cv2.imshow("Hailo-8L - Vision Robot 4WD", display_frame)

                    if cv2.waitKey(1) & 0xFF == 13:  # Tecla ENTER para cerrar
                        break

            finally:
                cam.stop()
                cv2.destroyAllWindows()
                print("--> Recursos liberados exitosamente.")