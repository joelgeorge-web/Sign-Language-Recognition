import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Dataset directory
dataset_dir = 'asl_dataset'

# Load images and labels
images = []
labels = []

# Iterate over the dataset directory
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        # Check if the file is an image
        if file.endswith('.jpg') or file.endswith('.jpeg'):
            # Load the image
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Print image path for debugging
            print('Loaded image:', image_path)
            
            images.append(image)

            # Extract the label from the directory name
            label = os.path.basename(root)
            labels.append(label)

# Convert the images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Check if any images were loaded
if len(images) == 0:
    print('No images found in the dataset directory. Please check the directory path and ensure it contains valid image files.')
    exit()

# Print the number of loaded images
print('Number of loaded images:', len(images))

# Flatten the images
images_flattened = images.reshape(len(images), -1)
print('done')

# Convert labels to integer representation
label_dict = {label: i for i, label in enumerate(np.unique(labels))}
labels_encoded = np.array([label_dict[label] for label in labels])
print('done')

# Create an SVM model
model = cv2.ml.SVM_create()
print('created model')

# Set SVM parameters
model.setType(cv2.ml.SVM_C_SVC)
model.setKernel(cv2.ml.SVM_LINEAR)
print('set parametters')

# Train the SVM model
model.train(images_flattened.astype(np.float32), cv2.ml.ROW_SAMPLE, labels_encoded.astype(np.int32))
print('trained')

# Save the trained model in XML format
model.save('svm_model.xml')
print('saved')

# Load the SVM model
model = cv2.ml.SVM_load('svm_model.xml')

# Define the class labels
class_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
                19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print('Failed to open the camera.')
    exit()

# Set the dimensions for resizing the captured image
width = 200
height = 200

# Read and process frames from the camera
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly
    if ret:
        # Resize the frame
        frame_resized = cv2.resize(frame, (width, height))

        # Convert the resized frame to grayscale
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Flatten the grayscale image
        frame_flattened = frame_gray.flatten()

        # Reshape the flattened image to match the training data
        frame_reshaped = frame_flattened.reshape(1, -1)

        # Predict the sign
        _, result = model.predict(frame_reshaped)

        # Get the predicted class label
        predicted_label = class_labels[int(result[0, 0])]

        # Display the frame with the predicted label
        cv2.putText(frame_resized, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera', frame_resized)

        # Check if the user wants to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()