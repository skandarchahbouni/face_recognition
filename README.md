# Face Recognition using OpenCV 👤

This project implements face recognition using OpenCV. It consists of two main components: training a face recognition model and making predictions using the trained model.

## Model Training 🛠️

The training code extracts features and labels from images in a specified directory, utilizing the Haar Cascade Classifier for face detection. The trained model is saved for future use to avoid re-execution.

## Face Recognition Implementation 🔍

The face recognition code loads the trained model to make predictions on random images from a directory of face images. It evaluates the model's accuracy using true and predicted labels.

### Features ✨

- Random face prediction from a directory 📸
- Accuracy evaluation using confusion matrix and accuracy score 📊

## Usage 📖

1. **Train the Model**: Run the training code to create and save the trained model.
2. **Make Predictions**: Run the prediction code to test the model with random images.
