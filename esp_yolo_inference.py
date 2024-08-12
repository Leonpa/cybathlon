import cv2
import torch
from torchvision import transforms
import numpy as np
import time

# Load the YOLOv7 model
model = torch.hub.load('/Users/leonpaletta/Coding/Cybathlon/recognition/yolov7', 'custom', '/Users/leonpaletta/Coding/Cybathlon/recognition/best.pt',
                       source='local')
model.eval()

# Video stream URL
stream_url = "http://192.168.4.1/"

# Attempt to connect to the video stream with retries
cap = None
for i in range(5):  # Retry 5 times
    cap = cv2.VideoCapture(stream_url)
    if cap.isOpened():
        break
    else:
        print(f"Attempt {i + 1}: Couldn't connect to stream. Retrying...")
        time.sleep(5)  # Wait 5 seconds before retrying

if not cap.isOpened():
    print("Failed to connect to the stream after multiple attempts.")
else:
    print("Connected to the stream.")

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
        print("Failed to grab frame")
        break

    # Convert frame to YOLOv7 format
    img = transform(frame).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        predictions = model(img)[0]  # Get the first batch's predictions

    print(f'The content of predictions is {predictions} and the shape of predictions is {predictions.shape}')

    # Parse predictions
    for idx, pred in enumerate(predictions):
        print(f'Prediction {idx}: {pred}')

        # Extract bounding box coordinates and confidences
        x1, y1, x2, y2 = pred[0:4].tolist()
        print(f'The thing is {pred[4]}')
        object_conf = pred[4].item()
        class_scores = pred[5:]  # Assuming the rest are class confidence scores

        # Get the class with the highest confidence
        class_conf, cls = torch.max(class_scores, dim=0)

        print(f'Bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}')
        print(f'Object confidence: {object_conf}, Class confidence: {class_conf.item()}, Class index: {cls.item()}')

        # Check if the object confidence is above the threshold
        if object_conf > 0.5:  # Confidence threshold (adjust as needed)
            label = int(cls.item())  # Get the label corresponding to the highest class score
            confidence = object_conf  # Use the scalar value of object_conf

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Label: {label} Conf: {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print(f'Skipped prediction with low confidence: {object_conf}')

    # Display the resulting frame
    cv2.imshow('YOLOv7 Inference', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
