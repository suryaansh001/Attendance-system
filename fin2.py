from ultralytics import YOLO
import pyttsx3
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime
import random
import time

# Initialize the TTS engine
engine = pyttsx3.init()

# Set properties (optional)
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)

# Load the YOLO model for pose detection
pose_model = YOLO('yolov8n-pose.pt')

# Load the TensorFlow model and label encoder
class_model = load_model('model.keras')
with open('/home/surya/Desktop/proj/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Define a function to preprocess image for the classification model
def preprocess_image(img, image_size=(224, 224)):
    img = cv2.resize(img, image_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Generate a task specifying which hand to raise
def generate_task():
    hand = ['left', 'right']
    current_hand = random.choice(hand)
    task = f"Raise {current_hand} hand"
    return current_hand, task

# Helper function to mark attendance
def markAttendance(name):
    with open('Attendance.csv', 'a') as f:  # Open in append mode
        now = datetime.now()
        dtString = now.strftime("%m/%d/%Y,%H:%M:%S")
        f.write(f'{name},{dtString}\n')

# Open the camera (0 for default camera)
cap = cv2.VideoCapture(0)
time.sleep(2)
# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Generate the task to display
required_hand, task = generate_task()
print("Task:", task)

# Variable to track if hand raise was detected
hand_raised_detected = False
detected_hand = None

# Main loop to read frames from the camera
while True:
    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:
        break  # If frame not read successfully, exit the loop

    # Perform pose detection on the captured frame
    results = pose_model(frame)

    # Loop through each detection result
    for result in results:
        # Check if keypoints are detected in the result
        if result.keypoints is not None:
            for keypoints in result.keypoints:
                keypoint_data = keypoints.data[0]  # Extract keypoints data

                # Extract coordinates and confidence scores for keypoints
                right_shoulder = keypoint_data[6]  # Right shoulder (x, y, confidence)
                left_shoulder = keypoint_data[5]   # Left shoulder (x, y, confidence)
                right_elbow = keypoint_data[8]     # Right elbow (x, y, confidence)
                left_elbow = keypoint_data[7]      # Left elbow (x, y, confidence)
                
                # Check if the confidence scores of shoulders and elbows are above 0.5
                if right_shoulder[2] > 0.5 or left_shoulder[2] > 0.5:
                    if right_elbow[2] > 0.5 or left_elbow[2] > 0.5:
                        # Check if either elbow is above the corresponding shoulder
                        if (right_elbow[1] < right_shoulder[1] and right_elbow[1] > 0):
                            # Display "Right hand raised" on the frame in green color
                            cv2.putText(frame, "Right hand raised", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            detected_hand = "right"
                            hand_raised_detected = True
                            engine.say("Right hand raised")
                            engine.runAndWait()
                            break

                        elif (left_elbow[1] < left_shoulder[1] and left_elbow[1] > 0):
                            # Display "Left hand raised" on the frame in green color
                            cv2.putText(frame, "Left hand raised", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            detected_hand = "left"
                            hand_raised_detected = True
                            engine.say("Left hand raised")
                            engine.runAndWait()
                            break

                        else:
                            # Display "Hand is not raised" on the frame in red color
                            cv2.putText(frame, "Hand is not raised", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Break the outer loop if hand is raised
        if hand_raised_detected:
            break

    # If hand was detected and it matches the required hand, classify the frame
    if hand_raised_detected and detected_hand == required_hand:
        # Preprocess the image for the classification model
        img_resized = preprocess_image(frame)

        # Predict using the classification model
        predictions = class_model.predict(img_resized)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = label_encoder.inverse_transform(predicted_class)[0]

        # Mark attendance and display the label
        markAttendance(predicted_label.upper())
        cv2.putText(frame, f"{predicted_label.upper()} - Task Matched", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        engine.say(f"Attendance marked for: {predicted_label.upper()}")
        engine.runAndWait()

        # Reset the task for the next candidate
        required_hand, task = generate_task()
        print("New Task:", task)
        hand_raised_detected = False

    else:
        # If hand raise does not match the task, display the task
        cv2.putText(frame, f"Task: {task}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show the frame with annotations in a window named 'frame'
    cv2.imshow('frame', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

