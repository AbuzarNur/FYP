import cv2
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt
from operator import itemgetter

'''
Abuzar Nur - FYP 2018 - Final Batch
'''

green_average = list()
frames = 0

vid = 'old3.mov'
cap = cv2.VideoCapture(vid)
NoF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(NoF)
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #high accuracy/slow


while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret:
		frames = frames + 1
		print(str(frames) + " out of " + str(NoF))
		# Our operations on the frame come here
		gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

		# Detect faces in the image
		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=4,
			minSize=(20, 20),
			flags = cv2.CASCADE_SCALE_IMAGE
		)

		area = list()

		if np.ndim(faces) > 1:
			for (x, y, w, h) in faces:
				area.append(w*h)
			area_max = max(area)
			area_index = area.index(area_max)
			face = faces[area_index]
		elif len(faces) == 0 :
			break
		else:
			print(faces)
			face = faces[0]

		x_scale = 0.23
		x_centre = round(face[0] + face[2] / 2)
		x_left = round(x_centre - face[2] * x_scale)
		x_right = round(x_centre + face[2] * x_scale)

		y_scale = 0.07
		y_centre = round(face[1] + face[3] / 2)
		y_centre = round(y_centre - face[3] * 0.35)
		y_top = round(y_centre - face[2] * y_scale)
		y_bottom = round(y_centre + face[2] * y_scale)

		ROI = [[int(x_left), int(y_top), int(x_right - x_left), int(y_bottom - y_top)]]

		# Draw a rectangle around the faces
		for (x, y, w, h) in ROI:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

			green = frame[y:y+h, x:x+w, 1]

		green_average.append(np.mean(green))

		# Display the resulting frame
		# cv2.imshow("frame", frame)
		# cv2.waitKey(0)
		if cv2.waitKey(2) & 0xFF == ord('q') or frames == NoF:
			# When everything done, release the capture
			cap.release()
			# cv2.destroyWindow("frame")
			cv2.destroyAllWindows()

			break


# frames = frames - 1
frames_range = np.arange(0, frames, 1)

print("Plotting Green Channel")
plt.figure(1)
plt.plot(frames_range, green_average)
plt.title("Raw Green Channel")
plt.savefig("Plots/green_raw.png")

# input("\nPRESS RETURN TO PROCEED\n")

print("Creating Filter")
fps = 30
bpm_range = [40/60, 200/60]

[b, a] = butter(2, [2*bpm_range[0]/fps, 2*bpm_range[1]/fps], btype = 'band')
green_filtered = lfilter(b, a, green_average)

print("Plotting Filtered Channel")
plt.figure(2)
plt.plot(frames_range, green_filtered)
plt.title("Filtered Green Channel")
plt.savefig("Plots/green_filtered.png")

signal = green_filtered[5*fps+1:np.size(green_filtered)]
signal_range = np.arange(5*fps+1, frames, 1)

print("Plotting Signal")
plt.figure(3)
plt.plot(signal_range, signal)
plt.title("Green Signal")
plt.savefig("Plots/green_signal.png")

print("Fourier Transforming")

window = 10
T_sample = round(fps * 0.1)
num_samples = round(window * fps)
num_bpm_samples = np.floor((np.size(signal) - num_samples)/T_sample)
bpm = list()
count = 0
padding = round(fps * (60 - window))

# print(num_samples)
for i in range(1, int(num_bpm_samples)-1,1):
	start = (i-1)*T_sample + 1
	signal_cutoff = signal[start : start+num_samples]

	signal_hann = np.multiply(signal_cutoff, np.hanning(np.size(signal_cutoff)))
	signal_DFT = np.abs(np.fft.fft(signal_hann))
	signal_padded = np.abs(np.fft.fft(np.pad(signal_hann, (0, padding), 'constant', constant_values=(0,0))))

	lower_padded = int(np.floor(bpm_range[0] * (np.size(signal_padded)/fps)) + 1)
	upper_padded = int(np.ceil(bpm_range[1] * (np.size(signal_padded)/fps)) + 1)
	bounds_padded = range(lower_padded,upper_padded,1)

	signal_padded_bounded = signal_padded[lower_padded:upper_padded]

	location, _ = find_peaks(signal_padded_bounded)
	peak = itemgetter(location)(signal_padded_bounded)
	peak_max = round(max(peak), 5)
	peak_rounded = [round(i, 5) for i in peak]
	peak_location = peak_rounded.index(peak_max)
	peak_index = bounds_padded[location[peak_location]]


	bpm.append((peak_index - 1) * (60 * fps / np.size(signal_padded)))
	count = count + 1
	print(str(count) + " out of " + str(num_bpm_samples))

print("Creating BPM plot")
time = np.multiply((range(0, count)), ((np.size(signal)/fps)/(num_bpm_samples-1)))
average_bpm = np.mean(bpm)
print(average_bpm)

plt.figure(4)
plt.plot(time, bpm)
plt.title("Heart Rate")
plt.savefig("Plots/bpm_plot.png")

print("Showing all graphs")
# plt.show()
input("\nPRESS RETURN TO EXIT\n")



