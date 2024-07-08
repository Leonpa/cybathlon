import os
import cv2
import numpy as np
import time
from picamera2 import Picamera2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.color_classify import classify_white
import mediapipe as mp
import threading


def process_detections(detection_result):
    for detection in detection_result:
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))

        label = detection.categories[0].category_name
        confidence = detection.categories[0].score

        # Print the bounding box coordinates and the detected object's class
        print(f"Detected {label} with confidence {confidence:.2f}")
        print(f"Bounding box: Start({start_point}), End({end_point})")


def main():
    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration(main={"size": (1600, 1200)})  # Set resolution to 1600x1200
    picam2.configure(camera_config)
    picam2.start()

    base_options = python.BaseOptions(model_asset_path='models/model_2class.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    time.sleep(0.1)  # Allow the camera to warm up

    # Initialize variables for framerate calculation
    frame_count = 0
    start_time = time.time()
    fps = 0

    def capture_and_process():
        nonlocal frame_count, start_time, fps

        while True:
            frame_start_time = time.time()

            frame = picam2.capture_array()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = detector.detect(mp_image)
            classified_detections = classify_white(image_rgb, detection_result.detections)
            process_detections(classified_detections)

            # Calculate the current framerate
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                print(f"Current FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()

            # Framerate limiter
            frame_end_time = time.time()
            frame_duration = frame_end_time - frame_start_time
            sleep_time = max(0, (1.0 / 10) - frame_duration)  # Limit to 10 FPS
            time.sleep(sleep_time)

    try:
        capture_thread = threading.Thread(target=capture_and_process)
        capture_thread.start()
        capture_thread.join()

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        picam2.close()


if __name__ == "__main__":
    main()
