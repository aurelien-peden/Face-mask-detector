import cv2

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # To predict using CPU

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from facedetector import FaceDetector
import numpy as np
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor = 0.6


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

        self.model = load_model('../face_mask.model')
        self.face_detector = FaceDetector(
            './haarcascade_frontalface_default.xml')

    def __del__(self):
        self.video.release()

    def _resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))
            
        resized = cv2.resize(image, dim, interpolation=inter)

        return resized

    def get_frame(self):
        (ret, frame) = self.video.read()

        frame = self._resize(frame, width=450)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.dstack([frame, frame, frame])
        face_rects = self.face_detector.detect(frame, scaleFactor=1.2,
                                               minNeighbors=3, minSize=(40, 40))

        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_image = frame[y:y+h, x:x+w]

            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_image = cv2.resize(face_image, (224, 224))

            face_image = face_image/255.0

            face_image = img_to_array(face_image)
            face_image = np.expand_dims(face_image, axis=0)

            (mask, without_mask) = self.model.predict(face_image)[0]

            if mask > without_mask:
                print("Mask! :", mask)
                color = (0, 255, 0)
                cv2.putText(frame, "Mask : {}%".format(mask), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            else:
                color = (0, 0, 255)
                cv2.putText(frame, "No mask : {}%".format(without_mask), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                print("without mask ! : ", without_mask)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
