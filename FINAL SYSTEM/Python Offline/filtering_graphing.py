'''
Abuzar Nur - Nihal Noor - FYP 2018 - Final Batch
'''
# TODO Check within cv2 - understand how it works

# Import all neccssary libraries
import cv2
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from operator import itemgetter
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# Empty variables
signal_average = list()
frames = 0

# Importing video file
vid = 'Amy2.avi'

try:
	signal_average = list(np.load("{}_signal_average.npy".format(vid)))
except IOError:
	cap = cv2.VideoCapture(vid)
	NoF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	# fps = int(cap.get(cv2.CAP_PROP_FPS))
	# print(fps)

	# Create the haar cascade to detect face
	faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

	while(True):
		ret, frame = cap.read() #ret = T/F if frame is received, frame = individual frame
		if ret:
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
			x_scale = 0.08
			x_centre = round(face[0] + face[2] / 2)
			x_left = round(x_centre - face[2] * x_scale)
			x_right = round(x_centre + face[2] * x_scale)

			y_scale = 0.05
			y_centre = round(face[1] + face[3] / 2)
			y_centre = round(y_centre - face[3] * 0.32)
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
			frames = frames + 1
			# Display the resulting frame
			# cv2.imshow("frame", frame)
			# cv2.waitKey(0)
			if cv2.waitKey(2) & 0xFF == ord('q') or frames == NoF:
				# When everything done, release the capture
				cap.release()
				# cv2.destroyWindow("frame")
				cv2.destroyAllWindows()

				break
		else:
			break
	np.save("{}_signal_average".format(vid),signal_average)

frames = len(signal_average)
frames_range = np.arange(0, frames, 1)

print("Plotting Green Channel fft")
plt.figure(1)
plt.plot(frames_range,signal_average)
plt.title("Raw Green Channel")
plt.savefig("Plots/green_raw.png")

# input("\nPRESS RETURN TO PROCEED\n")
###############################################################
print("Creating Filter")
fps = 30
bpm_range = [60/60, 150/60]

[b, a] = butter(2, [1*bpm_range[0]/fps, 2*bpm_range[1]/fps], btype = 'band') #creating butterworth filter coefficients
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
T_sample = 1
num_samples = round(window * fps)
num_bpm_samples = np.floor((np.size(signal) - num_samples)/T_sample)
bpm = list()
count = 0
padding = round(fps * (60 - window)) # length of padding

for i in range(1, int(num_bpm_samples)-1,1):
	start = (i-1)*T_sample + 1 # starting value of sliding window
	signal_cutoff = signal[start : (start+num_samples)] # sliding window range

	signal_hann = np.multiply(signal_cutoff, np.hanning(np.size(signal_cutoff))) # smoothing the window out
	# signal_DFT = np.abs(np.fft.fft(signal_hann)) # retrieving the absoloute value of the fft of the smoothed signal window
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

# outlier removal
window_size = 10
window_factor = 10
for i in np.arange(0, len(bpm)-1, 1):
	if (abs(bpm[i] - np.median(bpm)) > (0.1 * np.mean(bpm))) or (abs(bpm[i] - bpm[i + 1]) > (0.1 * np.mean(bpm))):
		if i >= 0 and i <= window_size:
			bpm[i] = (np.mean(bpm[0:i + window_size*window_factor]) + np.median(bpm[0:i + window_size*window_factor])) / 2
		elif i >= len(bpm)-window_size-1 and i <= len(bpm)-1:
			bpm[i] = (np.mean(bpm[i-window_size*window_factor:i]) + np.median(bpm[i-window_size*window_factor:i])) / 2
		else:
			bpm[i] = (np.mean(bpm[i - window_size:i + window_size]) + np.median(bpm[i - window_size:i + window_size])) / 2

perc_diff = 0.1
for i in np.arange(0, len(bpm)-1, 1):
	if (abs(bpm[i] - np.median(bpm)) > (perc_diff*np.mean(bpm))) or (abs(bpm[i] - bpm[i+1]) > (perc_diff*np.mean(bpm))):
		bpm[i] = (np.mean(bpm)+np.median(bpm))/2

print("Creating BPM plot")
time = np.multiply((range(0, count)), ((np.size(signal)/fps)/(num_bpm_samples-1))) # creating a time vector
average_bpm = np.mean(bpm)
print(average_bpm)

# smoothing graph
x = np.linspace(time.min(),time.max(), 1000)
itp = interp1d(time, bpm, kind = 'linear')
y = savgol_filter(itp(x), 101, 3)

plt.figure(4)
plt.rc('grid', linestyle=":", color='lightgray')
plt.grid()
# plt.plot(time, bpm)
plt.plot(x, y, color='lime')
plt.ylim((40, 120))
plt.xlabel('Time (s)')
plt.ylabel('Beats per Minute (bpm)')
plt.legend(["Average BPM = {}".format(round(average_bpm,1))])
# plt.title("Heart Rate: {}".format(vid))
plt.title("Heartrate")
plt.savefig("Plots/bpm_plot.png")
plt.savefig("Plots/{}.png".format(vid))


# plt.show()
input("\nPRESS RETURN TO EXIT\n")
