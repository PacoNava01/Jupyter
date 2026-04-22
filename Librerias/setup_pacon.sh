#!/bin/bash

echo " Actualizando sistema..."
sudo apt update && sudo apt upgrade -y

echo " Instalando cámara + OpenCV + GPIO..."
sudo apt install -y \
libcamera-apps \
python3-picamera2 \
python3-opencv \
python3-gpiozero \
python3-lgpio\
python3-numpy \
libatlas-base-dev \
libopenblas-dev



echo " Instalando herramientas base..."
sudo apt install -y \
python3-pip \
python3-venv \
git

echo " Creando entorno virtual .pacon..."
if [ -d ".pacon" ]; then
    echo "⚠️ .pacon ya existe, omitiendo creación"
else
    python3 -m venv .pacon --system-site-packages
fi

echo " Activando entorno..."
source .pacon/bin/activate

echo " Actualizando pip..."
pip install --upgrade pip setuptools wheel

echo " Instalando librerías Adafruit..."
pip install \
Adafruit-Blinka \
adafruit-circuitpython-servokit \
adafruit-circuitpython-pca9685

echo " Verificando instalaciones..."

python3 -c "import cv2; print('OpenCV OK')"
python3 -c "from gpiozero import LED; print('GPIOZERO OK')"

echo " Setup completo listo"
echo " Activa el entorno con: source .pacon/bin/activate"
