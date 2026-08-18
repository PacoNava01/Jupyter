import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams)

# 1. Definición de tus 3 clases personalizadas
CUSTOM_CLASSES = ["red", "green", "blue"]

# Colores BGR para dibujar cada clase: Rojo (B=0,G=0,R=255), Verde (B=0,G=255,R=0), Azul (B=255,G=0,R=0)
CLASS_COLORS = {
    0: (0, 0, 255),    # Red
    1: (0, 255, 0),    # Green
    2: (255, 0, 0)     # Blue
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

# 2. Cargar tu modelo compilado .hef
hef_path = "Tesis-Proyecto/Yolo_dir/yolov8n.hef"
hef = HEF(hef_path)

input_info = hef.get_input_vstream_infos()[0]
input_h, input_w = input_info.shape[0], input_info.shape[1]
input_name = input_info.name

# Inspeccionar nombre de capas de salida
output_infos = hef.get_output_vstream_infos()
print(f"Capas de salida detectadas en el HEF: {[info.name for info in output_infos]}")

# 3. Configurar Hardware Hailo-8L
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

            CONF_THRESHOLD = 0.2  # Umbral de confianza
            print("Iniciando detección de colores RGB en vivo (Presiona ENTER para salir)...")

            try:
                while True:
                    frame = cam.capture_array() # Frame en RGB desde Picamera2
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    if frame is None:
                        break
                    # Frame para OpenCV (RGB a BGR)

                    h_orig, w_orig, _ = frame.shape

                    # Preprocesamiento hacia la NPU
                    resized_img = cv2.resize(frame, (input_w, input_h))
                    input_data = {input_name: np.expand_dims(resized_img, axis=0).astype(np.uint8)}

                    # Inferencia en el chip Hailo-8L
                    raw_results = infer_pipeline.infer(input_data)

                    # Frame para OpenCV (RGB a BGR)
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Desempaquetado de las salidas
                    for out_name, out_tensor in raw_results.items():
                        if isinstance(out_tensor, (list, np.ndarray)) and len(out_tensor) > 0:
                            detections_batch = out_tensor[0]

                            # Recorrer las clases detectadas de forma segura (sin np.array global)
                            for class_id, boxes_for_class in enumerate(detections_batch):
                                if class_id >= len(CUSTOM_CLASSES):
                                    break
                                
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

                                            label_name = CUSTOM_CLASSES[class_id]
                                            color_box = CLASS_COLORS.get(class_id, (0, 255, 0))
                                            caption = f"{label_name} {score:.2f}"

                                            # Dibujar cuadro delimitador y etiqueta
                                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color_box, 2)
                                            cv2.putText(display_frame, caption, (x1, max(y1 - 8, 15)),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

                    cv2.imshow("Hailo AI Kit - Deteccion RGB012", display_frame)

                    # Salir con la tecla Enter (13)
                    if cv2.waitKey(1) & 0xFF == 13:
                        break
            finally:
                cam.stop()
                cv2.destroyAllWindows()
                print("Cámara liberada y recursos cerrados.")