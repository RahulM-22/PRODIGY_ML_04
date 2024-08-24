import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define the gestures based on folder names
gestures = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']

def load_images(dataset_path):
    images = []
    labels = []
    # Traverse through the main folders (like '00', '01', etc.)
    for main_folder in os.listdir(dataset_path):
        main_folder_path = os.path.join(dataset_path, main_folder)
        if not os.path.isdir(main_folder_path):
            continue
        # Traverse through each gesture folder (like '01_palm')
        for gesture_folder in os.listdir(main_folder_path):
            gesture_path = os.path.join(main_folder_path, gesture_folder)
            # Extract the gesture label (like 'palm')
            gesture_name = gesture_folder.split('_')[-1]
            if gesture_name not in gestures:
                continue
            label = gestures.index(gesture_name)
            # Process each image in the gesture folder
            for img_file in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (64, 64))  # Resize to 64x64 pixels
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load dataset
dataset_path = 'D:/MP 1 FINAL/ML 3/leapGestRecog'
X, y = load_images(dataset_path)

# Normalize the images
X = X.astype('float32') / 255.0

# Convert labels to categorical
y = to_categorical(y, num_classes=len(gestures))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(gestures), activation='softmax')  # Output layer with softmax activation
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=32)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.2f}')
import cv2

cap = cv2.VideoCapture(0)  # Open webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for prediction
    img = cv2.resize(frame, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make a prediction
    pred = model.predict(img)
    gesture = gestures[np.argmax(pred)]
    
    # Display the result on the video feed
    cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
