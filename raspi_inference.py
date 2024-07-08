import os
import cv2
import numpy as np
import time
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter, load_delegate
from utils.color_classify import classify_white
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
    camera_config = picam2.create_preview_configuration(main={"size": (1600, 1200)})  # Set resolution to 1600x1200
    picam2.configure(camera_config)
    picam2.start()

    # Load TensorFlow Lite model with GPU delegate
    interpreter = Interpreter(
        model_path='models/model_2class.tflite',
        experimental_delegates=[load_delegate('libedgetpu.so.1')]  # Use GPU delegate
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def detect_objects(image):
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])

    time.sleep(0.1)  # Allow the camera to warm up

    def capture_and_process():
        while True:
            frame = picam2.capture_array()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(image_rgb, axis=0).astype(np.float32) / 255.0  # Normalize input

            start_time = time.time()
            detection_result = detect_objects(input_data)
            end_time = time.time()

            print(f"Time taken: {end_time - start_time:.2f} seconds")

            classified_detections = classify_white(image_rgb, detection_result)
            process_detections(classified_detections)

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
