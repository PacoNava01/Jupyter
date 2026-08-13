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
