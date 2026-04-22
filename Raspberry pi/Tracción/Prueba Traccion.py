print(" TEST INICIO")

# GPIOZERO
try:
    from gpiozero import LED
    print(" gpiozero OK")
except Exception as e:
    print("gpiozero ERROR:", e)

# OpenCV
try:
    import cv2
    print(" OpenCV OK:", cv2.__version__)
except Exception as e:
    print(" OpenCV ERROR:", e)

# NumPy
try:
    import numpy as np
    print(" numpy OK:", np.__version__)
except Exception as e:
    print(" numpy ERROR:", e)

# Adafruit Blinka
try:
    import board
    print(" Adafruit Blinka OK")
except Exception as e:
    print(" Adafruit ERROR:", e)

try:
    import picamera2
    print(" picamera2 OK")
except Exception as e:
    print(" picamera2 ERROR:", e)



print(" TEST FINAL")