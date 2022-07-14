import os
import cv2
import numpy as np

def create_train(dir_path):
    """

    :param dir_path:
    :return: tuple(features, labels)
    """
    features = []  # X_train
    labels = []  # y_train
    class_names = os.listdir(DIR)
    for person in class_names:
        person_path = os.path.join(dir_path, person)
        label = class_names.index(person)
        person_photos = os.listdir(person_path)
        for photo in person_photos:
            img_path = os.path.join(person_path, photo)
            gray = cv2.imread(img_path, 0)
            # grabbing the region of interest (the face from) using the haar cascade algorithm
            classifier = cv2.CascadeClassifier('haar_face.xml')
            faces_rect = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            # Sometimes no face is detected, and sometimes multiple face are detected
            for (x, y, w, h) in faces_rect:
                roi = gray[y:y + h, x:x + w]
                features.append(roi)
                labels.append(label)


    features = np.array(features, dtype='object')
    labels = np.array(labels)
    return features, labels, class_names

# this is taking a lot of time, so wel will save them to not execute each time
DIR = "Faces/train"
features, labels, class_names = create_train(dir_path=DIR)
np.save('features.npy', features)
np.save('labels.npy', labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')