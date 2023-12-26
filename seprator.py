import torch
from time import sleep
from picamera import PiCamera
from detector import Detector
from model import Encoder


class RobotSeparator:
    def __init__(self, model_path, device):
        self.model = Encoder().to(device)
        self.model.load_state_dict(torch.load(model_path))

    def separate(self, img):
        out = self.model(img)
        return out == 1

# Usage
pin = 26
camera = PiCamera()
detector = Detector(pin, camera)
separator = RobotSeparator('model_1.pt', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

while True:
    sleep(0.5) # time between each loop
    if detector.detect():
        img = detector.take_an_image(0.01, "sample")
        if separator.separate(img):
            #TO Do go to trash: motor in position 1
            pass
        else:
            #TO Do leave it:    motor in position 0
            pass
    else:
        pass
