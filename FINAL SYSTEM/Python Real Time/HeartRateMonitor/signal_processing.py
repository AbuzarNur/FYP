import numpy as np
import cv2
import collections
from scipy.signal import butter, lfilter

class getBPM(object):

    def __init__(self):

        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.fps = 30
        self.bpm = list()
        self.mode_bpm = list()
        self.signal_filtered = list()
        self.signal_raw = list()

        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])

        self.data_buffer = []
        self.buffer_size = 300

        self.find_faces = True
        self.bpm_range = [50 / 60, 150 / 60]
        [self.b, self.a] = butter(2, [1 * self.bpm_range[0] / self.fps, 2 * self.bpm_range[1] / self.fps],btype = 'band')  # creating butterworth filter coefficients

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

    def get_subface_coord(self):
        x, y, w, h = self.face_rect
        # scaling the detected face into forehead
        x_scale = 0.08
        x_centre = round(x+ w / 2)
        x_left = round(x_centre - w * x_scale)
        x_right = round(x_centre + w * x_scale)

        y_scale = 0.05
        y_centre = round(y + h / 2)
        y_centre = round(y_centre - h * 0.32)
        y_top = round(y_centre - w * y_scale)
        y_bottom = round(y_centre + w * y_scale)

        return [int(x_left),
                int(y_top),
                int(x_right - x_left),
                int(y_bottom - y_top)]

    def get_subface_means(self, coord):
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]
        r = np.mean(subframe[:, :, 0])
        g = np.mean(subframe[:, :, 1])
        b = np.mean(subframe[:, :, 2])

        # return (r + g + b) / 3.
        return g


    def run(self, cam):
        self.frame_out = self.frame_in
        self.gray =cv2.cvtColor(self.frame_in,cv2.COLOR_BGR2GRAY)
        col = (100, 255, 100)

        if self.find_faces:
            # cv2.putText(self.frame_out, "Press 'C' to change camera (current: %s)" % str(cam),(10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            cv2.putText(self.frame_out, "Press 'S' to lock face and begin",(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            cv2.putText(self.frame_out, "Press 'Esc' to quit",(10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            self.data_buffer = []
            self.bpm = []
            self.signal_filtered = []
            self.signal_raw = []
            self.mode_bpm = []

            detected = list(self.face_cascade.detectMultiScale(self.gray, scaleFactor = 1.1, minNeighbors = 4, minSize = (20, 20), flags = cv2.CASCADE_SCALE_IMAGE))

            if len(detected) > 0:
                detected.sort(key = lambda a: a[-1] * a[-2])
                if self.shift(detected[-1]) > 2:
                    self.face_rect = detected[-1]

            self.draw_rect(self.face_rect, col = (255, 0, 0))
            x, y, w, h = self.face_rect
            cv2.putText(self.frame_out, "Face", (x, y-3), cv2.FONT_HERSHEY_DUPLEX, 0.5, col)

            return

        detected = list(self.face_cascade.detectMultiScale(self.gray, scaleFactor = 1.3, minNeighbors = 4, minSize = (50, 50),flags = cv2.CASCADE_SCALE_IMAGE))

        if len(detected) > 0:
            detected.sort(key = lambda a: a[-1] * a[-2])
            if (self.shift(detected[-1]) > 2) & (self.shift(detected[-1]) < 5) :
                self.face_rect = detected[-1]

        forehead = self.get_subface_coord()
        self.draw_rect(forehead)
        x, y, w, h = forehead

        cv2.putText(self.frame_out, "Press 'S' to restart",(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col)
        cv2.putText(self.frame_out, "Press 'Esc' to quit",(10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col)

        rgb = self.get_subface_means(forehead)
        self.data_buffer.append(rgb)

        if len(self.data_buffer) > 450:
            self.signal_filtered = lfilter(self.b, self.a, self.data_buffer) # applying buterworth filter onto signal
            offset = 5
            self.signal_raw = self.signal_filtered[offset * self.fps:]


        if len(self.signal_raw) > self.buffer_size:
            signal = self.signal_raw[-self.buffer_size:]

            window = 8  # window size seconds
            T_sample = 1 # seconds
            num_frames = round(window * self.fps) # frames per window
            num_bpm_samples = np.floor((np.size(signal) - num_frames) / T_sample) # bpm samples per window

            padding = round(self.fps * (60 - window))  # length of padding in frames

            for i in range(1, int(num_bpm_samples)-1, 1):
                start = (i - 1) * T_sample + 1  # starting value of sliding window
                signal_cutoff = signal[start: (start + num_frames)]  # sliding window range

                signal_hann = np.multiply(signal_cutoff, np.hanning(np.size(signal_cutoff)))  # smoothing the window out
                signal_padded = np.abs(np.fft.fft(np.pad(signal_hann, (0, padding), 'constant', constant_values = (0, 0))))  # applying post-padding

                lower_padded = int(np.floor(self.bpm_range[0] * (np.size(signal_padded) / self.fps)) + 1)  # lower bound
                upper_padded = int(np.ceil(self.bpm_range[1] * (np.size(signal_padded) / self.fps)) + 1)  # upper bound
                bounds_padded = np.arange(lower_padded, upper_padded, 1)  # range of bounded signal

                signal_padded_bounded = signal_padded[lower_padded:upper_padded]  # restricting the padded signal

                peak_location = np.argmax(signal_padded_bounded)  # finds the max value in the window and returns the index
                peak_index = bounds_padded[int(peak_location)]  # uses the index to return the bounded frequency

                self.bpm.append(peak_index)

        perc_diff = 0.1
        if len(self.bpm) > 100:
            if abs(self.bpm[-1] - np.median(self.bpm)) > (perc_diff * np.mean(self.bpm)):
                self.bpm.append((np.mean(self.bpm) + np.median(self.bpm)) / 2)

        if len(self.bpm) > 0:
            idx = np.where(np.greater(self.bpm, 55) & np.less(self.bpm, 110))
            bpm = np.array(self.bpm)
            counter = collections.Counter(np.round(bpm[idx[0]]))
            most_com = list(counter.most_common(3))
            print(most_com)

            num_values = 1000
            num_diff = 10
            if len(most_com) > 0:
                if (most_com[0][1] > num_values) and (most_com[1][1] > num_values) and (most_com[2][1] > num_values):
                    if (abs(most_com[0][0] - most_com[1][0]) < num_diff) and (abs(most_com[0][0] - most_com[2][0]) < num_diff) and (
                            abs(most_com[1][0] - most_com[2][0]) < num_diff):
                        self.mode_bpm.append(np.mean([most_com[0][0],most_com[1][0],most_com[2][0]]))

        if len(self.mode_bpm) > 0:
            text = "(BPM: %0.1f +/- %0.1f)" %(self.mode_bpm[-1], self.mode_bpm[-1]/10)
        elif len(self.bpm) > 0:
            text = "(bpm: %0.1f +/- %0.1f)" % (self.bpm[-1], self.bpm[-1] / 10)
        elif (450 - len(self.data_buffer)) > 0:
            text = "(wait: %0.1f )" % ((450 - len(self.data_buffer)) / self.fps)
        else:
            text = "calculating..."

        col = (0, 255, 0)
        cv2.putText(self.frame_out, text, (x-w*2, y-5), cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 1)