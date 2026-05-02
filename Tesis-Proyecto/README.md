🤖 Carro 4WD Autónomo con Visión por Computadora
Plataforma: Raspberry Pi + OpenCV + PyTorch
Este proyecto implementa el sistema de control de un vehículo terrestre de cuatro ruedas (4WD) que integra procesamiento de imágenes en tiempo real para navegación autónoma, detección de objetos y evasión de obstáculos.

📂 Estructura del Proyecto
```
nombre-del-proyecto/
│
├── data/                   # Modelos entrenados (.xml, .tflite, .pth) y datasets de prueba
├── docs/                   # Diagramas de conexión, esquemas de pines (Pinout) y bitácora
├── src/                    # CÓDIGO FUENTE PRINCIPAL
│   ├── hardware/           # Abstracción de componentes físicos
│   │   ├── motor_driver.py # Control de motores y cinemática diferencial
│   │   └── sensors.py      # Telemetría (Ultrasonido, IMU, IR)
│   ├── vision/             # Pipeline de procesamiento de imágenes
│   │   ├── detector.py     # Segmentación semántica y detección de objetos
│   │   └── camera.py       # Gestión de frames mediante hilos (Threading)
│   ├── utils/              # Herramientas de apoyo
│   │   └── helpers.py      # Transformaciones matemáticas y registro de logs
│   └── main.py             # Script de ejecución principal (Orquestador)
│

├── tests/                  # Pruebas unitarias para validación de hardware y visión
├── config.yaml             # Configuración centralizada de GPIO y parámetros de cámara
├── requirements.txt        # Dependencias (OpenCV, NumPy, PyTorch, RPi.GPIO)
└── README.md               # Documentación del proyecto
```


🛠️ Descripción de Módulos Clave
🏎️ Hardware & Cinemática (src/hardware/)
motor_driver.py: Implementa una capa de abstracción para el puente H (L298N) o controlador PWM (PCA9685). Permite comandos de alto nivel como robot.move_to(target_velocity) en lugar de manipular pines individuales.

👁️ Visión Artificial (src/vision/)
camera.py: Implementa captura de video asíncrona mediante Threading. Esto garantiza que el flujo de control del hardware no se bloquee mientras se procesan redes neuronales o filtros de imagen pesados.

detector.py: Contiene la lógica para la detección de líneas y objetos utilizando OpenCV (espacios de color HSV) o arquitecturas como CNNs para segmentación.

⚙️ Configuración (config.yaml)
Centraliza los parámetros del sistema para evitar "hardcoding":

GPIO Mapping: Asignación de pines para motores y sensores.

Camera Settings: Resolución, FPS y parámetros de calibración.

Vision Params: Umbrales de color y constantes de detección.

