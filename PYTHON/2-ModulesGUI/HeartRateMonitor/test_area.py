# import cv2
# import numpy as np
# import tkinter as tk
#
# class Camera(object):
#     def __init__(self, camera = 0):
#         self.camera = camera
#         self.prompt()
#
#     def capture_input(self):
#         if len(self.entry) == 0:
#             self.cap = cv2.VideoCapture(self.camera)
#         else:
#             self.cap = cv2.VideoCapture(self.entry)
#
#         self.valid = False
#         try:
#             resp = self.cap.read()
#             self.shape = resp[1].shape
#             self.valid = True
#         except:
#             self.shape = None
#
#         # tk.Tk.destroy(self)
#         print(self.valid)
#
#     def prompt(self):
#         root = tk.Tk()
#
#         label = tk.Label(root, text = "Enter video file (""file_name.format"") or blank for Webcam: ")
#         label.grid(row = 0, column = 0)
#         entry = tk.Entry(root)
#         entry.focus_force()
#         entry.grid(row = 0, column = 1)
#         self.entry = entry.get()
#
#         button = tk.Button(root, text = "Submit", command = self.capture_input())
#         button.grid(row = 1, column = 1)
#
#         root.mainloop()
#
#     def get_frame(self):
#         if self.valid:
#             ret,frame = self.cap.read()
#         else:
#             frame = np.ones((480,640,3), dtype = np.uint8)
#             col = (0,256,256)
#             cv2.putText(frame, "(Error: Capture not accessible)",
#                        (65,220), cv2.FONT_HERSHEY_PLAIN, 2, col)
#         return frame
#
#     def release(self):
#         self.cap.release()
#
# Camera(0)
#
