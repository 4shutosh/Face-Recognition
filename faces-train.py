import os
import numpy as np
from PIL import Image
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_DIR = os.path.join(BASE_DIR, "training_data")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # in order to use the face cascade class
recognizerLBPH = cv2.face.LBPHFaceRecognizer_create()

temp_id = 0
label_ids = {}
labels = []
train = []

for root, dirs, files, in os.walk(image_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(path)

            if label in label_ids:
                pass
            else:
                label_ids[label] = temp_id
                temp_id += 1

            id_ = label_ids[label]
            # print(label_ids)

            # labels.append(label)
            # train.append(path)
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array)

            for a, b, c, d in faces:
                face_part = image_array[b:b + d, a:a + c]
                train.append(face_part)
                labels.append(id_)

# print(labels)
# print(train)

# save the label_ids dictionary

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizerLBPH.train(train, np.array(labels))
recognizerLBPH.save("trainer.yml")
