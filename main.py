import os
import cv2
import random

DIR = "Faces/val"
class_names = os.listdir(DIR)

# Loading our trained model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

# Make predictions using our model (make sure it's a 2 dimensions image)
random_class = random.choice(class_names)
random_img = random.choice(os.listdir(os.path.join(DIR, random_class)))
img = cv2.imread(DIR + "/" + random_class + "/" + random_img, 0)

classifier = cv2.CascadeClassifier('haar_face.xml')
face_rect = classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in face_rect:
    roi = img[y:y+h, x:x+w]

label, confidence = face_recognizer.predict(roi)
print(confidence)
cv2.putText(img, text=class_names[label] + f"     {confidence:.2f}", org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=0, thickness=2)

cv2.imshow(random_class, img)
cv2.waitKey()

# -------------------

true_labels = []  # y_train
predicted_labels = []
class_names = os.listdir(DIR)
for person in class_names:
    person_path = os.path.join(DIR, person)
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
            true_labels.append(label)
            label_pred, confidence = face_recognizer.predict(roi)
            predicted_labels.append(label_pred)

print(len(true_labels), len(predicted_labels))

# Evaluation

from sklearn.metrics import accuracy_score
acc = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy : {acc}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix :")
print(cm)