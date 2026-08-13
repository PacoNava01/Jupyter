import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams)

# Clases por defecto de COCO (80 categorías)
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
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

def letterbox(img, new_shape=(640,640), color=(114,114,114)):
    """Resize manteniendo proporción y agregando padding."""
    shape = img.shape[:2]  # altura, ancho
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw /= 2; dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, dw, dh

# 1. Configuración del modelo y dispositivo
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
                print("No se pudo iniciar la cámara. Saliendo...")
                exit(1)

            print("Iniciando bucle de inferencia en la NPU...")
            CONF_THRESHOLD = 0.5

            try:
                while True:
                    frame = cam.capture_array()
                    if frame is None:
                        print("Error: Frame vacío de la cámara.")
                        break

                    h_orig, w_orig, _ = frame.shape

                    # Preprocesamiento con letterbox y normalización
                    resized_frame, r, dw, dh = letterbox(frame, (input_width, input_height))
                    normalized = resized_frame.astype(np.float32) / 255.0
                    input_data = {input_name: np.expand_dims(normalized, axis=0)}

                    raw_results = infer_pipeline.infer(input_data)

                    # Mostrar las claves disponibles (solo la primera vez)
                    print("Claves de salida disponibles:", raw_results.keys())

                    display_frame = cv2.rotate(frame, cv2.ROTATE_180)

                    # Buscar automáticamente la primera clave de salida
                    output_key = list(raw_results.keys())[0]
                    detections = raw_results[output_key]

                    for det in detections:
                        if len(det) >= 6:
                            ymin, xmin, ymax, xmax, score, class_id = det[:6]
                            if score >= CONF_THRESHOLD:
                                x1 = int((xmin * w_orig))
                                y1 = int((ymin * h_orig))
                                x2 = int((xmax * w_orig))
                                y2 = int((ymax * h_orig))

                                label_idx = int(class_id)
                                label = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f"ID:{label_idx}"
                                caption = f"{label} {score:.2f}"

                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                (w_txt, h_txt), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(display_frame, (x1, max(y1 - 20, 0)), (x1 + w_txt, max(y1, 20)), (0, 255, 0), -1)
                                cv2.putText(display_frame, caption, (x1, max(y1 - 5, 15)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    cv2.imshow("Hailo AI Kit - Deteccion YOLOv8", display_frame)
                    if cv2.waitKey(1) & 0xFF == 13:
                        break
            finally:
                cam.stop()
                cv2.destroyAllWindows()
                print("Cámara liberada y ventanas cerradas.")
