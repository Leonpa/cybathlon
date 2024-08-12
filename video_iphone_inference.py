import cv2
import torch
from torchvision import transforms
import numpy as np
import time

# Load the YOLOv7 model
model = torch.hub.load('/Users/leonpaletta/Coding/Cybathlon/recognition/yolov7', 'custom', '/Users/leonpaletta/Coding/Cybathlon/recognition/best.pt', source='local')
model.eval()

# Video file path
video_path = "/Users/leonpaletta/Coding/Cybathlon/recognition/data/videos/test2.mp4"

# Attempt to open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Failed to open video file: {video_path}")
else:
    print(f"Successfully opened video file: {video_path}")

# Define image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Inference loop
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to grab frame")
        break

    # Convert frame to YOLOv7 format
    img = transform(frame).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        predictions = model(img)[0]  # Get the first batch's predictions

    # Parse predictions
    for pred in predictions:
        x1, y1, x2, y2 = pred[0:4]  # Bounding box coordinates
        object_conf = pred[4]  # Object confidence (single scalar)
        class_scores = pred[5:]  # Class confidence scores

        # Get the class with the highest confidence
        class_conf, cls = torch.max(class_scores, dim=0)

        # Check if the object confidence is above the threshold
        if object_conf.item() > 0.5:  # Confidence threshold (adjust as needed)
            label = int(cls.item())  # Get the label corresponding to the highest class score
            confidence = float(object_conf.item())

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Label: {label} Conf: {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv7 Inference', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
