import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams)

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

# 1. Configuración del modelo y dispositivo
hef_path = "Tesis-Proyecto/Yolo_dir/yolov8n.hef"
hef = HEF(hef_path)

params = VDevice.create_params()
with VDevice(params) as target:
    # Configurar el grupo de inferencia pasando 'hef' y 'configure_params'
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    # Definir parámetro de entrada y salida
    input_vstream_params = InputVStreamParams.make(network_group)
    output_vstream_params = OutputVStreamParams.make(network_group)

    # Obtener información de entrada de la red (ej. dimensiones)
    input_info = hef.get_input_stream_infos()[0]
    input_height, input_width = input_info.shape[0], input_info.shape[1]
    input_name = input_info.name

    # ACTIVAR EL GRUPO DE RED (Obligatorio para evitar errores de hardware)
    with network_group.activate(network_group_params):
        # Inicializar streams de datos
        with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
            
            # Inicializar la Raspberry Pi HQ Camera
            cam = init_cam()
            if cam is None:
                print("No se pudo iniciar la cámara. Saliendo...")
                exit(1)

            print("Iniciando bucle de inferencia en la NPU...")
            
            try:
                while True:
                    # Capturar frame desde Picamera2 (devuelve un array RGB en formato numpy)
                    frame = cam.capture_array()
                    if frame is None:
                        print("Error: Frame vacío de la cámara.")
                        break

                    # Pre-procesamiento: Redimensionar al tamaño que espera la red
                    resized_frame = cv2.resize(frame, (input_width, input_height))
                    
                    # Asegurar formato uint8 (estándar para YOLO en Hailo)
                    input_data = {input_name: np.expand_dims(resized_frame, axis=0).astype(np.uint8)}

                    # Inferencia en la NPU
                    raw_results = infer_pipeline.infer(input_data)

                    # TODO: Implementar post-procesamiento de las salidas (raw_results)

                    # Mostrar visualización (Picamera2 captura en RGB, OpenCV muestra bien en RGB con imshow si se prefiere, o BGR)
                    # Si los colores se ven extraños (azulados), descomenta la siguiente línea:
                    # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    cv2.imshow("Hailo AI Kit - Deteccion YOLOv8", frame)
                    
                    # Presiona 'Enter' (código 13) para salir
                    if cv2.waitKey(1) & 0xFF == 13:
                        break
            finally:
                # Asegurar que la cámara se cierre correctamente al salir
                cam.stop()
                cv2.destroyAllWindows()
                print("Cámara liberada y ventanas cerradas.")