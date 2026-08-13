import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams)

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
        print("Cámara inicializada correctamente.")
        return cam
    except Exception as e:
        print(f"Error al inicializar la cámara: {e}")
        return None

hef_path = "Tesis-Proyecto/Yolo_dir/yolov8n.hef"
hef = HEF(hef_path)

params = VDevice.create_params()
with VDevice(params) as target:
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_vstream_params = InputVStreamParams.make(network_group)
    output_vstream_params = OutputVStreamParams.make(network_group)

    input_info = hef.get_input_stream_infos()[0]
    input_height, input_width = input_info.shape[0], input_info.shape[1]
    input_name = input_info.name

    with network_group.activate(network_group_params):
        with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
            
            cam = init_cam()
            if cam is None:
                exit(1)

            print("Iniciando bucle de inferencia en la NPU...")
            CONF_THRESHOLD = 0.3  # Bajamos a 0.3 para asegurar capturar detecciones iniciales

            try:
                while True:
                    frame = cam.capture_array() # Frame en RGB
                    if frame is None:
                        break

                    h_orig, w_orig, _ = frame.shape

                    # Pre-procesamiento
                    resized_frame = cv2.resize(frame, (input_width, input_height))
                    input_data = {input_name: np.expand_dims(resized_frame, axis=0).astype(np.uint8)}

                    # Inferencia
                    raw_results = infer_pipeline.infer(input_data)

                    # Frame para visualización en OpenCV (convertimos a BGR)
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Procesamiento de la salida NMS
                    for key, detections in raw_results.items():
                        # detections suele ser una lista de listas por lote (batch)
                        for batch in detections:
                            for det in batch:
                                # Manejo seguro según el objeto devuelto por HailoRT
                                try:
                                    # Si Hailo devuelve el objeto nativo HailoDetection
                                    if hasattr(det, 'get_bbox'):
                                        bbox = det.get_bbox()
                                        ymin, xmin, ymax, xmax = bbox.ymin(), bbox.xmin(), bbox.ymax(), bbox.xmax()
                                        score = det.get_confidence()
                                        class_id = det.get_class_id()
                                    # Si devuelve una estructura/tupla de NumPy
                                    else:
                                        ymin, xmin, ymax, xmax, score, class_id = det[:6]
                                except Exception:
                                    continue

                                if score >= CONF_THRESHOLD:
                                    x1 = int(xmin * w_orig)
                                    y1 = int(ymin * h_orig)
                                    x2 = int(xmax * w_orig)
                                    y2 = int(ymax * h_orig)

                                    label_idx = int(class_id)
                                    label = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f"ID:{label_idx}"
                                    caption = f"{label} {score:.2f}"

                                    # Dibujar caja
                                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(display_frame, caption, (x1, max(y1 - 5, 15)), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    cv2.imshow("Hailo AI Kit - Deteccion YOLOv8", display_frame)
                    
                    if cv2.waitKey(1) & 0xFF == 13: # Enter
                        break
            finally:
                cam.stop()
                cv2.destroyAllWindows()
                print("Cámara liberada.")
