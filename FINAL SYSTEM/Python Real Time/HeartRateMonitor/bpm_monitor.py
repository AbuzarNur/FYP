from HeartRateMonitor.webcam import Camera
from HeartRateMonitor.signal_processing import getBPM
from cv2 import imshow, waitKey
import sys

class HeartRateMonitor(object):
    def __init__(self):
        self.cameras = []
        self.selected_cam = 0
        camera = Camera(camera = 0)
        if camera.valid or not len(self.cameras):
            self.cameras.append(camera)

        self.w = 0
        self.h = 0
        self.pressed = 0

        self.processor = getBPM()

        self.key_controls = {"s": self.toggle_search,
                             "c": self.toggle_cam}

    def toggle_cam(self):
        if len(self.cameras) > 1:
            self.processor.find_faces = True
            self.selected_cam += 1
            self.selected_cam = self.selected_cam % len(self.cameras)

    def toggle_search(self):
        state = self.processor.find_faces_toggle()
        print("Face Detection Lock-On: ", not state)

    def key_handler(self):
        self.pressed = waitKey(10) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("Exiting")
            for cam in self.cameras:
                cam.cam.release()
            sys.exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def main_loop(self):
        frame = self.cameras[self.selected_cam].get_frame()
        self.h, self.w, _c = frame.shape

        self.processor.frame_in = frame
        self.processor.run(self.selected_cam)
        output_frame = self.processor.frame_out
        imshow("Video Based Heart Rate Monitor (BETA)", output_frame)

        self.key_handler()

if __name__ == "__main__":
    App = HeartRateMonitor()
    while True:
        App.main_loop()
