import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import os

# Example: Adjust file paths
file_path = "C:/Users/Ayush/codes/.vscode/.vscode/HAND GESTURES/leapGestRecog/00/01_palm.png"

# Read image
image = cv2.imread(file_path)

# Check if the image is successfully loaded
if image is not None:
    # Your image processing code here
    print("Image loaded successfully.")
else:
    print(f"Unable to read image: {file_path}")
# Load the dataset
data_path = 'C:/Users/Ayush/codes/.vscode/.vscode/HAND GESTURES/leapGestRecog'
gestures = os.listdir(data_path)
gestures.sort()

images = []
labels = []

for gesture_id, gesture in enumerate(gestures):
    gesture_path = os.path.join(data_path, gesture)
    if os.path.isdir(gesture_path):  # Ensure it's a directory
        for image_name in os.listdir(gesture_path):
            image_path = os.path.join(gesture_path, image_name)
            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Unable to read image: {image_path}")
                    continue
                # Check image size before resizing
                print(f"Image size before resizing: {image.shape}")
                image = cv2.resize(image, (128, 128))
                images.append(image)
                labels.append(gesture_id)
            except Exception as e:
                print(f"Error processing image: {image_path}\nError: {e}")

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize pixel values to be between 0 and 1
images = images / 255.0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(gestures), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train.reshape(-1, 128, 128, 1), y_train, epochs=10, validation_data=(X_test.reshape(-1, 128, 128, 1), y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test.reshape(-1, 128, 128, 1), y_test)
print(f'Test accuracy: {test_acc}')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
