import cv2
import argparse
from facedetector import FaceDetector

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # To predict using CPU


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to where the image file reside")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
if h > 1080:
    image = cv2.resize(image, (600, 800))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fd = FaceDetector("./haarcascade_frontalface_default.xml")
faceRects = fd.detect(gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
print(f'I found {len(faceRects)} face(s)')

model = load_model('./face_mask.model')

face_images = []
for (x, y, w, h) in faceRects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    face_image = image[y:y+h, x:x+w]

    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = cv2.resize(face_image, (224, 224))

    face_image = face_image/255.0

    face_image = img_to_array(face_image)
    face_image = np.expand_dims(face_image, axis=0)

    face_images.append(face_image)

    (mask, without_mask) = model.predict(face_image)[0]

    if mask > without_mask:
        print("Mask! :", mask)
        color = (0, 255, 0)
        cv2.putText(image, "Mask : {}%".format(mask), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    else:
        color = (0, 0, 255)
        cv2.putText(image, "No mask : {}%".format(without_mask), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        print("without mask ! : ", without_mask)

cv2.imshow("Faces", image)
cv2.waitKey(0)