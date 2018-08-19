import cv2
import numpy as np

'''
Abuzar Nur - FYP 2018 - Webcam Testing
'''

cap = cv2.VideoCapture(0);
print(cap)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    size = faces.shape
    area = list()

    if size[0] > 1:
        for (x, y, w, h) in faces:
            area.append(w * h)

        area_max = max(area)
        area_index = area.index(area_max)
        face = faces[area_index]
    else:
        face = faces[0]

    x_scale = 0.19
    x_centre = round(face[0] + face[2] / 2)
    x_left = round(x_centre - face[2] * x_scale)
    x_right = round(x_centre + face[2] * x_scale)

    y_scale = 0.062
    y_centre = round(face[1] + face[3] / 2)
    y_centre = round(y_centre - face[3] * 0.35)
    y_top = round(y_centre - face[2] * y_scale)
    y_bottom = round(y_centre + face[2] * y_scale)

    ROI = [[int(x_left), int(y_top), int(x_right - x_left), int(y_bottom - y_top)]]

    # Draw a rectangle around the faces
    for (x, y, w, h) in ROI:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("frame", frame)
    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(faces)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()