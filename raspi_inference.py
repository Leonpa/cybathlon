import os
import cv2
import numpy as np
import time
from picamera2 import Picamera2, Preview
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.color_classify import classify_white
import mediapipe as mp

# Set the QT_QPA_PLATFORM_PLUGIN_PATH environment variable
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/qt/plugins'


def visualize(image, detection_result):
    for detection in detection_result:
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))

        if "-soft" in detection.categories[0].category_name:
            color = (255, 0, 0)  # Red color for -soft objects
        else:
            color = (0, 255, 0)  # Green color for -hard objects

        thickness = 2

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


def main():
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration()
    picam2.configure(camera_config)
    picam2.start_preview(Preview.QTGL)
    picam2.start()

    base_options = python.BaseOptions(model_asset_path='models/model_2class.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    time.sleep(0.1)  # Allow the camera to warm up

    try:
        while True:
            frame = picam2.capture_array()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = detector.detect(mp_image)
            classified_detections = classify_white(image_rgb, detection_result.detections)
            frame_copy = np.copy(image_rgb)
            annotated_frame = visualize(frame_copy, classified_detections)
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Frame", annotated_frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        picam2.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
