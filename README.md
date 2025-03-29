**Smart Attendance System: YOLOv8 Pose Detection and CNN Face Recognition**

**Overview** The Smart Attendance System utilizes YOLOv8 for pose detection and a Convolutional Neural Network (CNN) for face recognition to automate and streamline the attendance process, ensuring accurate and efficient tracking.

**Key Components Pose Detection with YOLOv8**
**
Pose Detection:** YOLOv8 is employed to detect specific poses, such as a hand-raising gesture, which confirms student presence.



**Object Detection**:YOLOv8-pose is employed to detect if phone or any other device is not being used for cheating the system


**Face Recognition with CNN**

 1) **Model Training:** Train a CNN model to recognize and authenticate student faces based on a dataset of labeled student images.
  
  2)**Face Recognition:** The CNN model identifies and verifies student identities in real-time. Workflow


Attendance Marking:** If the student’s hand-raising gesture is detected and their face is recognized, attendance is marked.
**
Implementation Steps Data Preparation: Collect and label images for both pose detection and face recognition. 
Model Training: Develop and train YOLOv8 for pose detection and the CNN model for face recognition. I
ntegration: Combine YOLOv8 and the CNN model into a unified system. Testing and Deployment: Validate the system’s accuracy and reliability before deploying it.
Considerations Privacy: Ensure all student data is handled securely and in compliance with privacy regulations. 
Accuracy: Regularly update and fine-tune models to maintain high performance. 
