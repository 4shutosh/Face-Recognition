# OpenCV module
import cv2

# os module for reading training data directories and paths
import os

# numpy to convert python lists to numpy arrays as it is needed by OpenCV face recognizers
import numpy as np

import pickle

# list of names for trained images
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"personsName": 1}
with open("labels.pickle", 'rb') as f:
    labels = pickle.load(f)
    labelsNew = {v: k for k, v in labels.items()}

cap = cv2.VideoCapture(0)

# face detection
# harCascade method for face detection
while True:
    # image = cv2.imread("inra.jpg", 1)  # importing image
    ret, frame = cap.read()
    face_cascade = cv2.CascadeClassifier(
        "haarcascades/haarcascade_frontalface_default.xml")  # in order to use the face cascade class

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converting to gray image

    facesFound = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)  # identifying facesFound

    print(type(facesFound))
    print(facesFound)

    for a, b, c, d in facesFound:

        face_part_gray = gray[b:b + d, a:a + c]
        face_part_color = frame[b:b + d, a:a + c]

        output_image = "my-image.png"
        cv2.imwrite(output_image, face_part_gray)

        id_, confidence = recognizer.predict(face_part_gray)

        if 45 <= confidence >= 75:
            print(id_)
            print(labelsNew[id_])
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labelsNew[id_]
            color = (255, 255, 255)
            cv2.putText(frame, name, (a, b), font, 1, color, 2, cv2.LINE_AA)

        # width = a + c
        # height = b + d
        # (0,255,0) is green
        # 3 represents the thickness of the line
        # adding rectangle to the face
        img = cv2.rectangle(frame, (a, b), (a + c, b + d), (0, 255, 0), 2)

    resized_image = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[1] / 2)))
    cv2.imshow("me", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# function to detect face using OpenCV
# def detect_face(img):
#     # convert the test image to gray scale as opencv face detector expects gray images
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # load OpenCV face detector, I am using LBP which is fast
#     # there is also a more accurate but slow: Haar classifier
#     face_cascade = cv2.CascadeClassifier("face_model.xml")
#
#     # let's detect multiscale images(some images may be closer to camera than others)
#     # result is a list of faces
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
#
#     # if no faces are detected then return original img
#     if len(faces) == 0:
#         return None, None
#
#     # under the assumption that there will be only one face,
#     # extract the face area
#     (x, y, w, h) = faces[0]
#
#     # return only the face part of the image
#     return gray[y:y + w, x:x + h], faces[0]
#
#
# # face recognition
# def build_training_data(training_data):
#     dirs = os.listdir(training_data)
#
#     # list to hold all subject faces
#     faces = []
#     # list to hold labels for all subjects
#     labels = []
#
#     # let's go through each directory and read images within it
#     for dir_name in dirs:
#
#         # our subject directories start with letter 's' so
#         # ignore any non-relevant directories if any
#         if not dir_name.startswith("s"):
#             continue;
#
#         # ------STEP-2--------
#         # extract label number of subject from dir_name
#         # format of dir name = slabel
#         # , so removing letter 's' from dir_name will give us label
#         label = int(dir_name.replace("s", ""))
#
#         # build path of directory containing images for current subject subject
#         # sample subject_dir_path = "training-data/amitabh bachchan"
#         subject_dir_path = training_data + "/" + dir_name
#
#     # get the images names that are inside the given subject directory
#     subject_images_names = os.listdir(subject_dir_path)
#
#     # ------STEP-3--------
#     # go through each image name, read image,
#     # detect face and add face to list of faces
#     for image_name in subject_images_names:
#
#         # ignore system files like .DS_Store
#         if image_name.startswith("."):
#             continue;
#
#         # build image path
#         # sample image path = training-data/amitabh bachchan/1.pgm
#         image_path = subject_dir_path + "/" + image_name
#
#         # read image
#         image = cv2.imread(image_path)
#
#         # display an image window to show the image
#         cv2.imshow("Training on image...", image)
#         cv2.waitKey(100)
#
#         # detect face
#         face, rect = detect_face(image)
#
#         # ------STEP-4--------
#         # for the purpose of this tutorial
#         # we will ignore faces that are not detected
#         if face is not None:
#             # add face to list of faces
#             faces.append(face)
#             # add label for this face
#             labels.append(label)
#
#             cv2.destroyAllWindows()
#             cv2.waitKey(1)
#             cv2.destroyAllWindows()
#
#     return faces, labels
#
#
# # let's first prepare our training data
# # data will be in two lists of same size
# # one list will contain all the faces
# # and the other list will contain respective labels for each face
# print("Preparing data...")
# faces, labels = build_training_data("training_data")
# print("Data prepared")
#
# # print total faces and labels
# print("Total faces: ", len(faces))
# print("Total labels: ", len(labels))
#
# # create our LBPH face recognizer
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# # train our face recognizer of our training faces
# face_recognizer.train(faces, np.array(labels))
#
#
# # function to draw rectangle on image
# # according to given (x, y) coordinates and
# # given width and height
# def draw_rectangle(img, rect):
#     (x, y, w, h) = rect
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#
# # function to draw text on give image starting from
# # passed (x, y) coordinates.
# def draw_text(img, text, x, y):
#     cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
#
#
# # this function recognizes the person in image passed
# # and draws a rectangle around detected face with name of the
# # subject
# def recognize(test_img):
#     # make a copy of the image as we don't want to change original image
#     if test_img is None:
#         return
#
#     img = test_img.copy()
#     # detect face from the image
#     face, rect = detect_face(img)
#
#     # predict the image using our face recognizer
#     label = face_recognizer.predict(face)
#     # get name of respective label returned by face recognizer
#     label_text = subjects[label[0]]
#
#     # draw a rectangle around face detected
#     draw_rectangle(img, rect)
#     # draw name of predicted person
#     draw_text(img, label_text, rect[0], rect[1] - 5)
#     print("recognition successful")
#     return img
#
#
# print("Recognizing images...")
#
# # load test images
# test_img1 = cv2.imread("test1.png")
# # test_img2 = cv2.imread("test-data/test2.jpg")
#
# # perform a prediction
# predicted_img1 = recognize(test_img1)
# # predicted_img2 = recognize(test_img2)
# print("Prediction complete")
#
# # display both images
# cv2.imshow(subjects[1], predicted_img1)
# # cv2.imshow(subjects[2], predicted_img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
