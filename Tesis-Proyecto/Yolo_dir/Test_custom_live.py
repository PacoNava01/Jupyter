import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams)

# Clases del dataset RGB012
CUSTOM_CLASSES = ["red", "green", "blue"]
CLASS_COLORS = {
    0: (0, 0, 255),    # Red (BGR)
    1: (0, 255, 0),    # Green (BGR)
    2: (255, 0, 0)     # Blue (BGR)
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

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_grid_strides(img_size=640, strides=(8, 16, 32)):
    grid_points, stride_tensor = [], []
    for s in strides:
        h_grid = img_size // s
        w_grid = img_size // s
        x, y = np.meshgrid(np.arange(w_grid), np.arange(h_grid))
        grid_points.append(np.stack([x.flatten(), y.flatten()], axis=-1))
        stride_tensor.append(np.full((h_grid * w_grid, 1), s))
    return np.vstack(grid_points), np.vstack(stride_tensor)

# 1. Configuración de modelo
hef_path = "Tesis-Proyecto/Yolo_dir/best_rgb012.hef"
hef = HEF(hef_path)

input_info = hef.get_input_vstream_infos()[0]
input_h, input_w = input_info.shape[0], input_info.shape[1]
input_name = input_info.name

# Generar cuadrículas de anclas
anchors, strides = generate_grid_strides(img_size=input_w)

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

            CONF_THRESHOLD = 0.4
            IOU_THRESHOLD = 0.45
            print("Iniciando detección con decodificación de anclas (Presiona ENTER para salir)...")

            try:
                while True:
                    frame = cam.capture_array()
                    if frame is None:
                        break
                    
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    h_orig, w_orig, _ = frame.shape

                    # 1. FIX DE COLORES: Asumimos que frame viene en BGR por OpenCV
                    display_frame = frame.copy() # OpenCV usa BGR nativamente para mostrar
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # YOLO necesita RGB

                    # Inferencia (Usando el frame RGB)
                    resized_img = cv2.resize(rgb_frame, (input_w, input_h))
                    input_data = {input_name: np.expand_dims(resized_img, axis=0).astype(np.uint8)}
                    raw_results = infer_pipeline.infer(input_data)

                    # 2. FIX DE TENSORES: Separar y ORDENAR por resolución espacial
                    cls_outputs_raw = []
                    box_outputs_raw = []

                    for name, tensor in raw_results.items():
                        t = tensor[0]
                        if t.shape[-1] == 3:
                            cls_outputs_raw.append(t)
                        elif t.shape[-1] in (16, 64):
                            box_outputs_raw.append(t)

                    # ¡CRÍTICO! Ordenar de mayor a menor resolución (80x80 -> 40x40 -> 20x20)
                    cls_outputs_raw.sort(key=lambda x: x.shape[0], reverse=True)
                    box_outputs_raw.sort(key=lambda x: x.shape[0], reverse=True)

                    cls_outputs, box_outputs = [], []

                    # Procesar en el orden correcto
                    for t_cls, t_box in zip(cls_outputs_raw, box_outputs_raw):
                        cls_outputs.append(t_cls.reshape(-1, 3))

                        # Decodificación DFL
                        dfl_scores = t_box.reshape(-1, 4, t_box.shape[-1] // 4)
                        exp_dfl = np.exp(dfl_scores - np.max(dfl_scores, axis=-1, keepdims=True))
                        dfl_weights = exp_dfl / np.sum(exp_dfl, axis=-1, keepdims=True)
                        integrated_box = np.sum(dfl_weights * np.arange(t_box.shape[-1] // 4), axis=-1)
                        box_outputs.append(integrated_box)

                    if cls_outputs and box_outputs:
                        all_cls = sigmoid(np.vstack(cls_outputs))
                        all_boxes = np.vstack(box_outputs)

                        # Decodificación de coordenadas relativas [lt, rb] a [x, y, w, h]
                        x1 = (anchors[:, 0] - all_boxes[:, 0]) * strides[:, 0]
                        y1 = (anchors[:, 1] - all_boxes[:, 1]) * strides[:, 0]
                        x2 = (anchors[:, 0] + all_boxes[:, 2]) * strides[:, 0]
                        y2 = (anchors[:, 1] + all_boxes[:, 3]) * strides[:, 0]

                        boxes = np.column_stack([
                            x1 * (w_orig / input_w),
                            y1 * (h_orig / input_h),
                            (x2 - x1) * (w_orig / input_w),
                            (y2 - y1) * (h_orig / input_h)
                        ]).astype(int)

                        # Supresión de No Máximos (NMS)
                        for class_id in range(len(CUSTOM_CLASSES)):
                            scores = all_cls[:, class_id]
                            mask = scores >= CONF_THRESHOLD
                            if np.any(mask):
                                filtered_boxes = boxes[mask].tolist()
                                filtered_scores = scores[mask].tolist()

                                indices = cv2.dnn.NMSBoxes(
                                    filtered_boxes, filtered_scores, CONF_THRESHOLD, IOU_THRESHOLD
                                )

                                for idx in indices:
                                    i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
                                    bx, by, bw, bh = filtered_boxes[i]
                                    score = filtered_scores[i]
                                    label_name = CUSTOM_CLASSES[class_id]
                                    color = CLASS_COLORS[class_id]

                                    cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), color, 2)
                                    cv2.putText(display_frame, f"{label_name} {score:.2f}",
                                                (bx, max(by - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5, color, 2)

                    cv2.imshow("Hailo AI Kit - Deteccion RGB012", display_frame)

                    if cv2.waitKey(1) & 0xFF == 13: # Enter para salir
                        break
            finally:
                cam.stop()
                cv2.destroyAllWindows()
                print("Cámara liberada.")