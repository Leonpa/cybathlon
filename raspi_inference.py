import cv2
import numpy as np
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.color_classify import classify_white
import mediapipe as mp


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
    base_options = python.BaseOptions(model_asset_path='model.tflite')
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


def main():
    # Initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # Create an ObjectDetector object with your custom model
    base_options = python.BaseOptions(model_asset_path='model.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    time.sleep(0.1)  # Allow the camera to warm up

    # Capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array

        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the frame to a format suitable for the detector
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Detect objects in the frame
        detection_result = detector.detect(mp_image)

        # Classify the color characteristics in the bounding boxes
        classified_detections = classify_white(image_rgb, detection_result.detections)

        # Visualize the classified detections
        frame_copy = np.copy(image_rgb)
        annotated_frame = visualize(frame_copy, classified_detections)

        # Convert annotated frame back to BGR for displaying
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Display the frame
        cv2.imshow("Frame", annotated_frame_bgr)

        # Clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # If the 'q' key is pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    camera.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
