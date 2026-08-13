from hailo_platform import HEF

# 1. Cargar el archivo compilado
hef_path = "Tesis-Proyecto/Yolo_dir/yolov8n.hef"
hef = HEF(hef_path)

# 2. Inspeccionar la entrada esperada (Input VStream)
input_info = hef.get_input_vstream_infos()[0]
print("--- ENTRADA DEL MODELO ---")
print(f"Nombre de la capa de entrada: {input_info.name}")
print(f"Dimensiones esperadas (Alto, Ancho, Canales): {input_info.shape}")
print(f"Tipo de dato esperado: {input_info.format.type}")

# 3. Inspeccionar la salida generada (Output VStreams)
output_infos = hef.get_output_vstream_infos()
print("\n--- SALIDA(S) DEL MODELO ---")
for info in output_infos:
    print(f"Nombre de capa de salida: {info.name}")
    print(f"Dimensiones devueltas: {info.shape}")

'''
(.pacon) pacon@Pacon:~/Jupyter $ /home/pacon/Jupyter/Librerias/.pacon/bin/python /home/pacon/Jupyter/Tesis-Proyecto/Yolo_dir/Yolo_test.py
--- ENTRADA DEL MODELO ---
Nombre de la capa de entrada: yolov8n/input_layer1
Dimensiones esperadas (Alto, Ancho, Canales): (640, 640, 3)
Tipo de dato esperado: FormatType.UINT8

--- SALIDA(S) DEL MODELO ---
Nombre de capa de salida: yolov8n/yolov8_nms_postprocess
Dimensiones devueltas: (80, 5, 100)

'''