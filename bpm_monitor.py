from HeartRateMonitor.webcam import Camera
from HeartRateMonitor.signal_processing import findFaceGetPulse
from HeartRateMonitor.gui import plotXY

from cv2 import moveWindow, imshow, waitKey, destroyWindow
import argparse
import numpy as np
import datetime
import sys

class getPulseApp(object):
    def __init__(self):
        self.cameras = []
        self.selected_cam = 0
        camera = Camera(camera = 0)
        if camera.valid or not len(self.cameras):
            self.cameras.append(camera)

        self.w = 0
        self.h = 0
        self.pressed = 0

        self.processor = findFaceGetPulse()

        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        self.key_controls = {"s": self.toggle_search,
                             "d": self.toggle_display_plot,
                             "c": self.toggle_cam,
                             "f": self.write_csv}

    def toggle_cam(self):
        if len(self.cameras) > 1:
            self.processor.find_faces = True
            self.bpm_plot = False
            destroyWindow(self.plot_title)
            self.selected_cam += 1
            self.selected_cam = self.selected_cam % len(self.cameras)

    def write_csv(self):
        fn = "Webcam-pulse" + str(datetime.datetime.now())
        fn = fn.replace(":", "_").replace(".", "_")
        data = np.vstack((self.processor.times, self.processor.samples)).T
        np.savetxt(fn + ".csv", data, delimiter=',')
        print("Writing csv")


    def toggle_search(self):
        state = self.processor.find_faces_toggle()
        print("face detection lock =", not state)

    def toggle_display_plot(self):
        if self.bpm_plot:
            print("bpm plot disabled")
            self.bpm_plot = False
            destroyWindow(self.plot_title)
        else:
            print("bpm plot enabled")
            if self.processor.find_faces:
                self.toggle_search()
            self.bpm_plot = True
            self.make_bpm_plot()
            moveWindow(self.plot_title, self.w, 0)

    def make_bpm_plot(self):
        plotXY([[self.processor.times,self.processor.samples], [self.processor.freqs,self.processor.fft]],
               labels = [False, True],
               showmax = [False, "bpm"],
               label_ndigits = [0, 0],
               showmax_digits = [0, 1],
               skip = [3, 3],
               name = self.plot_title,
               bg = self.processor.slices[0])

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
        imshow("Processed", output_frame)

        if self.bpm_plot:
            self.make_bpm_plot()

        self.key_handler()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Webcam pulse detector.')
    args = parser.parse_args()

    App = getPulseApp()
    while True:
        App.main_loop()
