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
from queue import Queue
import json
import socket

DISCOVERY_PORT = 5001  # Port for discovery messages


def discover_server():
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    udp_socket.settimeout(5)

    message = b'DISCOVER_SERVER'
    udp_socket.sendto(message, ('<broadcast>', DISCOVERY_PORT))

    try:
        data, server_address = udp_socket.recvfrom(1024)
        if data.decode('utf-8') == 'SERVER_HERE':
            return server_address[0]
    except socket.timeout:
        print("Server discovery timed out.")
        return None


def process_detections(detection_result, client_socket):
    for detection in detection_result:
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))

        label = detection.categories[0].category_name
        confidence = detection.categories[0].score

        center_x = int((bbox.origin_x + bbox.width) / 2)
        center_y = int((bbox.origin_y + bbox.height) / 2)

        data = {
            "label": label,
            "confidence": confidence,
            "start_point": start_point,
            "end_point": end_point,
            "center": (center_x, center_y)
        }

        client_socket.sendall(json.dumps(data).encode('utf-8'))

        print(f"Detected {label} with confidence {confidence:.2f}")
        print(f"Bounding box: x ({center_x}), y ({center_y})")


def main():
    # Discover server IP
    server_ip = discover_server()
    if not server_ip:
        print("Server not found!")
        return

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, 5000))  # Replace with server's IP and port

    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(main={"size": (1600, 1200)})  # Set resolution to 1600x1200
    picam2.configure(camera_config)
    picam2.start()

    base_options = python.BaseOptions(model_asset_path='models/model_2class.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    time.sleep(0.1)  # Allow the camera to warm up

    frame_queue = Queue(maxsize=1)

    def capture_frames():
        while True:
            frame = picam2.capture_array()
            if not frame_queue.full():
                frame_queue.put(frame)

    def process_frames():
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                detection_result = detector.detect(mp_image)
                classified_detections = classify_white(image_rgb, detection_result.detections)
                process_detections(classified_detections, client_socket)

    try:
        capture_thread = threading.Thread(target=capture_frames)
        process_thread = threading.Thread(target=process_frames)
        capture_thread.start()
        process_thread.start()
        capture_thread.join()
        process_thread.join()

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        picam2.close()
        client_socket.close()


if __name__ == "__main__":
    main()
