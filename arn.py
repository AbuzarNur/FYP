import numpy as np
import time
import cv2
import pylab

class findFaceGetPulse(object):

    def __init__(self):

        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.fps = 0
        self.buffer_size = 250
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpm = 0

        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13

        self.find_faces = True

        self.data_buffer = []
        self.times = []
        self.freqs = []
        self.fft = []

    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        return self.find_faces

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift

    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def get_subface_means(self, coord):
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]
        r = np.mean(subframe[:, :, 0])
        g = np.mean(subframe[:, :, 1])
        b = np.mean(subframe[:, :, 2])

        # return (r + g + b) / 3.
        return g

    def plot(self):
        data = np.array(self.data_buffer).T
        np.savetxt("data.dat", data)
        np.savetxt("times.dat", self.times)
        freqs = 60. * self.freqs
        idx = np.where((freqs > 50) & (freqs < 180))

        pylab.figure()
        n = data.shape[0]
        for k in range(n):
            pylab.subplot(n, 1, k + 1)
            pylab.plot(self.times, data[k])
        pylab.savefig("data.png")

        pylab.figure()
        for k in range(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(freqs[idx], self.fft[k][idx])
        pylab.savefig("data_fft.png")
        quit()

    def run(self, cam):
        self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,cv2.COLOR_BGR2GRAY))
        col = (100, 255, 100)

        if self.find_faces:
            # cv2.putText(self.frame_out, "Press 'C' to change camera (current: %s)" % str(cam),(10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            cv2.putText(self.frame_out, "Press 'S' to lock face and begin",(10, 50), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            cv2.putText(self.frame_out, "Press 'Esc' to quit",(10, 75), cv2.FONT_HERSHEY_PLAIN, 1.25, col)

            self.data_buffer = []
            self.times = []

            detected = list(self.face_cascade.detectMultiScale(self.gray, scaleFactor = 1.2, minNeighbors = 4, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE))

            if len(detected) > 0:
                detected.sort(key = lambda a: a[-1] * a[-2])
                if self.shift(detected[-1]) > 2:
                    self.face_rect = detected[-1]

            self.draw_rect(self.face_rect, col = (255, 0, 0))
            x, y, w, h = self.face_rect
            cv2.putText(self.frame_out, "Face", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)

            return

        detected = list(self.face_cascade.detectMultiScale(self.gray,scaleFactor = 1.2,minNeighbors = 4,minSize = (30, 30),flags = cv2.CASCADE_SCALE_IMAGE))

        if len(detected) > 0:
            detected.sort(key = lambda a: a[-1] * a[-2])
            if self.shift(detected[-1]) > 2:
                self.face_rect = detected[-1]

        forehead = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
        self.draw_rect(forehead)
        x, y, w, h = forehead

        cv2.putText(self.frame_out, "Press 'C' to change camera (current: %s)" % str(cam),(10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
        cv2.putText(self.frame_out, "Press 'S' to restart",(10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "Press 'D' to toggle data plot",(10, 75), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        # cv2.putText(self.frame_out, "Press 'F' to save csv", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "Press 'Esc' to quit",(10, 125), cv2.FONT_HERSHEY_PLAIN, 1.5, col)

        vals = self.get_subface_means(forehead)
###########
        self.data_buffer.append(vals)
        L = len(self.data_buffer)
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size

        processed = np.array(self.data_buffer)
        self.samples = processed
        if L > 10:
            self.output_dim = processed.shape[0]

            self.fps = float(L) / (self.times[-1] - self.times[0])
            # print(self.fps)
            even_times = np.linspace(self.times[0], self.times[-1], L)
            interpolated = np.interp(even_times, self.times, processed)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)

            self.fft = np.abs(np.fft.rfft(interpolated))
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)

            freqs = 60. * self.freqs
            idx = np.where((freqs > 50) & (freqs < 180))

            self.freqs = freqs[idx]

            self.fft = self.fft[idx]
            idx_max = np.argmax(self.fft)

            self.bpm = self.freqs[idx_max]

            gap = (self.buffer_size - L) / self.fps

            if gap:
                text = "(estimate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
            else:
                text = "(estimate: %0.1f bpm)" % (self.bpm)

            cv2.putText(self.frame_out, text, (int(x - w / 2), int(y)), cv2.FONT_HERSHEY_PLAIN, 1, col)