import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/Users/leonpaletta/Coding/Cybathlon/recognition/models/coco_ssd_mobilenet_v1_1/detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to run inference on a frame
def run_inference(frame):
    # Assuming input tensor is 300x300 for MobileNet SSD
    height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

    return boxes, classes, scores

# Flag to switch between camera and video file
USE_CAMERA = False
VIDEO_FILE_PATH = '/Users/leonpaletta/Coding/Cybathlon/recognition/data/test2.mp4'  # Specify your video file path here

# Initialize video source, camera if USE_CAMERA is True, video file if False
cap = cv2.VideoCapture(0 if USE_CAMERA else VIDEO_FILE_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, classes, scores = run_inference(frame)

    # Process the results
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1], ymin * frame.shape[0], ymax * frame.shape[0])
            # Draw bounding box
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

    # Display output
    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
