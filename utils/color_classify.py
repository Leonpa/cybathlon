import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


def classify_white(image, detections):
    classified_detections = []
    for detection in detections:
        bbox = detection.bounding_box
        x_min, y_min = int(bbox.origin_x), int(bbox.origin_y)
        x_max, y_max = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)

        # Extract the bounding box region
        bbox_region = image[y_min:y_max, x_min:x_max]

        # Calculate the average intensity of the region
        avg_intensity = np.mean(bbox_region)

        classified_detections.append((detection, avg_intensity))

    # Determine the threshold for classification based on intensity
    threshold = 170

    # Classify detections into 'white' (hard) and 'non-white' (soft)
    for i in range(len(classified_detections)):
        detection, avg_intensity = classified_detections[i]
        if avg_intensity > threshold:  # Higher intensity means white (hard)
            detection.categories[0].category_name += "-hard"
        else:
            detection.categories[0].category_name += "-soft"

    return [detection for detection, _ in classified_detections]


def visualize(image, detection_result):
    for detection in detection_result:
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))

        # Determine the color based on the category name
        if "-soft" in detection.categories[0].category_name:
            color = (255, 0, 0)  # Red color for -soft objects
        else:
            color = (0, 255, 0)  # Green color for -hard objects

        thickness = 2

        # Draw a circle for specific objects and a rectangle for others
        if detection.categories[0].category_name in ["cat_2-hard", "cat_3-soft"]:
            center = (int((start_point[0] + end_point[0]) / 2), int((start_point[1] + end_point[1]) / 2))
            radius = int(((end_point[0] - start_point[0]) + (end_point[1] - start_point[1])) / 4)
            image = cv2.circle(image, center, radius, color, thickness)
        else:
            image = cv2.rectangle(image, start_point, end_point, color, thickness)

        label = detection.categories[0].category_name
        confidence = detection.categories[0].score
        text = f"{label}: {confidence:.2f}"
        image = cv2.putText(image, text, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def visualize_video(input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Create an ObjectDetector object with your custom model
    base_options = python.BaseOptions(model_asset_path='models/model_2class.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a format suitable for the detector
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect objects in the frame
        detection_result = detector.detect(mp_image)

        # Classify the color characteristics in the bounding boxes
        classified_detections = classify_white(frame_rgb, detection_result.detections)

        # Visualize the classified detections
        frame_copy = np.copy(frame_rgb)
        annotated_frame = visualize(frame_copy, classified_detections)

        # Convert annotated frame back to BGR for saving
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Write the annotated frame to the output video
        out.write(annotated_frame_bgr)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
