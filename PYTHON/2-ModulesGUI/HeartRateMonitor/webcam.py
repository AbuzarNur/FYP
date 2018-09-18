import cv2
import numpy as np

class Camera(object):
    def __init__(self, camera = 0):
        self.cam = cv2.VideoCapture(camera)
        self.cam.set(3, 640)  # width
        self.cam.set(4, 480)  # height
        self.cam.set(10, 127)  # brightness     min: 0   , max: 255 , increment:1
        self.cam.set(11, 127)  # contrast       min: 0   , max: 255 , increment:1
        self.cam.set(12, 255)  # saturation     min: 0   , max: 255 , increment:1
        # self.cam.set(13, 13)  # hue
        # self.cam.set(14, 64)  # gain           min: 0   , max: 127 , increment:1
        # self.cam.set(15, -3)  # exposure       min: -7  , max: -1  , increment:1
        # self.cam.set(17, 5000)  # white_balance  min: 4000, max: 7000, increment:1
        # self.cam.set(28, 0)  # focus          min: 0   , max: 255 , increment:5
        # self.cam = cv2.VideoCapture('NN57.avi')
        print(int(self.cam.get(cv2.CAP_PROP_FPS)))
        self.valid = False
        try:
            resp = self.cam.read()
            self.shape = resp[1].shape
            self.valid = True
        except:
            self.shape = None

    def get_frame(self):
        if self.valid:
            _,frame = self.cam.read()
            print(int(self.cam.get(cv2.CAP_PROP_FPS)))
        else:
            frame = np.ones((480,640,3), dtype = np.uint8)
            col = (0,255,255)
            cv2.putText(frame, "(Error: Camera not accessible)",(65,220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return frame

    def release(self):
        self.cam.release()
