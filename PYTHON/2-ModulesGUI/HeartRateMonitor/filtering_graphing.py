'''
Abuzar Nur - Nihal Noor - FYP 2018 - Final Batch
'''
# TODO Check within cv2 - understand how it works

# Import all neccssary libraries
import cv2
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt
from imutils.video import FPS
from imutils.video import FileVideoStream
import time
from operator import itemgetter

# Empty variables
signal_average = list()
frames = 0

# Importing video file
vid = 'old3.mov'
cap = cv2.VideoCapture(vid)
NoF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# Faster method
print("[INFO] starting video file thread...")
fvs = FileVideoStream(vid).start()
time.sleep(1.0)
fps = FPS().start()

# Create the haar cascade to detect face
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while fvs.more():
	frame = fvs.read()
	print("Queue Size: {}".format(fvs.Q.qsize()))
	frames = frames + 1
	print(str(frames) + " out of " + str(NoF))
	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=4,
		minSize=(20, 20),
		flags = cv2.CASCADE_SCALE_IMAGE
	)

	area = list() #creating empty list for now

	# make sure only the largest face in the frame is used
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

	# scaling the detected face into forehead
	x_scale = 0.20
	x_centre = round(face[0] + face[2] / 2)
	x_left = round(x_centre - face[2] * x_scale)
	x_right = round(x_centre + face[2] * x_scale)

	y_scale = 0.05
	y_centre = round(face[1] + face[3] / 2)
	y_centre = round(y_centre - face[3] * 0.38)
	y_top = round(y_centre - face[2] * y_scale)
	y_bottom = round(y_centre + face[2] * y_scale)

	# creating ROI with parameters {x, y, w, h}
	ROI = [[int(x_left), int(y_top), int(x_right - x_left), int(y_bottom - y_top)]]

	# Draw a rectangle around the faces
	for (x, y, w, h) in ROI:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		# capturing indiviudal rgb values from the ROI
		red = frame[y:y+h, x:x+w, 0]
		green = frame[y:y+h, x:x+w, 1]
		blue = frame[y:y+h, x:x+w, 2]

	# creating an average of the rgb channels
	red_average = np.mean(red)
	green_average = np.mean(green)
	blue_average = np.mean(blue)

	rgb = np.concatenate((green_average,red_average), axis=None) # putting together all the wanted channels

	signal_average.append(np.mean(green_average)) # using the average of rgb channels and creating a signal vector

	# Display the resulting frame
	# cv2.imshow("frame", frame)
	# cv2.waitKey(1)
	fps.update()

# When everything done, release the capture
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
fvs.stop()

# frames = frames - 1
frames_range = np.arange(0, frames, 1)

print("Plotting Green Channel")
plt.figure(1)
plt.plot(frames_range, signal_average)
plt.title("Raw Green Channel")
plt.savefig("Plots/green_raw.png")

# input("\nPRESS RETURN TO PROCEED\n")

print("Creating Filter")
fps = 30
bpm_range = [40/60, 180/60]

[b, a] = butter(2, [2*bpm_range[0]/fps, 2*bpm_range[1]/fps], btype = 'band') #creating butterworth filter coefficients
signal_filtered = lfilter(b, a, signal_average) # applying buterworth filter onto signal

print("Plotting Filtered Channel")
plt.figure(2)
plt.plot(frames_range, signal_filtered)
plt.title("Filtered Green Channel")
plt.savefig("Plots/green_filtered.png")

# cutting off initial frames where signal has not settled
offset = 5
signal = signal_filtered[offset*fps+1 : np.size(signal_filtered)]
signal_range = np.arange(offset*fps+1, frames, 1)

print("Plotting Signal")
plt.figure(3)
plt.plot(signal_range, signal)
plt.title("Green Signal")
plt.savefig("Plots/green_signal.png")

print("Fourier Transforming")
window = 10 # window size
T_sample = round(fps * 0.1) # sample period
num_samples = round(window * fps)
num_bpm_samples = np.floor((np.size(signal) - num_samples)/T_sample)
bpm = list()
count = 0
padding = round(fps * (60 - window)) # length of padding

for i in range(1, int(num_bpm_samples)-1,1):
	start = (i-1)*T_sample + 1 # starting value of sliding window
	signal_cutoff = signal[start : (start+num_samples)] # sliding window range

	signal_hann = np.multiply(signal_cutoff, np.hanning(np.size(signal_cutoff))) # smoothing the window out
	signal_DFT = np.abs(np.fft.fft(signal_hann)) # retrieving the absoloute value of the fft of the smoothed signal window
	signal_padded = np.abs(np.fft.fft(np.pad(signal_hann, (0, padding), 'constant', constant_values=(0,0)))) # applying post-padding

	lower_padded = int(np.floor(bpm_range[0] * (np.size(signal_padded)/fps)) + 1) # lower bound
	upper_padded = int(np.ceil(bpm_range[1] * (np.size(signal_padded)/fps)) + 1) # upper bound
	bounds_padded = np.arange(lower_padded,upper_padded,1) # range of bounded signal

	signal_padded_bounded = signal_padded[lower_padded:upper_padded] # restricting the padded signal

	# location, _ = find_peaks(signal_padded_bounded) # provides an index
	# peak = itemgetter(location)(signal_padded_bounded) #uses the index to get the indexed value
	# peak_max = round(max(peak), 5)
	# peak_rounded = [round(i, 5) for i in peak]
	# peak_location = peak_rounded.index(peak_max) #find the index where the matching value exists
	# peak_index = bounds_padded[location[peak_location]] #at the index

	peak_location_2 = np.argmax(signal_padded_bounded) # finds the max value in the window and returns the index
	peak_index_2 = bounds_padded[int(peak_location_2)] # uses the index to return the bounded frequency

	bpm.append((peak_index_2 - 1) * (60 * fps / np.size(signal_padded))) # appending the latest bpm value to the vector
	count = count + 1
	print(str(count) + " out of " + str(num_bpm_samples))

average_bpm = np.mean(bpm)
print(average_bpm)

for i in bpm:
	if abs(i - np.median(bpm)) > (0.1*average_bpm):
		i = np.median(bpm)

print("Creating BPM plot")
time = np.multiply((range(0, count)), ((np.size(signal)/fps)/(num_bpm_samples-1))) # creating a time vector

plt.figure(4)
plt.plot(time, bpm)
plt.ylim((40, 120))
plt.title("Heart Rate")
plt.savefig("Plots/bpm_plot.png")

# plt.show()
input("\nPRESS RETURN TO EXIT\n")
