import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import RPi.GPIO as GPIO

class Detector:
    def __init__(self, pin, camera):
        self.pin = pin
        self.camera = camera
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)

    def detect(self):
        return GPIO.input(self.pin) == GPIO.HIGH

    def take_an_image(self, a, b):
        a, b = a*32, b*32
        self.camera.resolution = (a, b)
        output = np.empty((a, b, 3), dtype=np.uint8)
        self.camera.capture(output,'rgb')
        img = cv2.rotate(output, cv2.ROTATE_180)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

