import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.color_classify import classify_white
import mediapipe as mp

# Class map
class_map = {
    "cat_2-hard": "Hard Cylinder",
    "cat_3-soft": "Soft Cylinder",
    "background-hard": "Hard Cube",
    "cat_1-soft": "Soft Cube"
}

def visualize_centers(image, detection_result):
    center_points = []
    for detection in detection_result:
        bbox = detection.bounding_box
        center_x = int(bbox.origin_x + bbox.width / 2)
        center_y = int(bbox.origin_y + bbox.height / 2)
        class_name = detection.categories[0].category_name
        class_label = class_map.get(class_name, "Unknown")

        center_points.append((center_x, center_y, class_label))

    # Create a white backdrop
    white_backdrop = np.ones_like(image) * 255

    for center_x, center_y, class_label in center_points:
        # Draw large red center points
        cv2.circle(white_backdrop, (center_x, center_y), 10, (0, 0, 255), -1)  # Red color for center points
        # Put the class label next to the dot
        cv2.putText(white_backdrop, class_label, (center_x + 15, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return white_backdrop, center_points

def optical_flow_stabilize(prev_frame, curr_frame, prev_gray, prev_points):
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    curr_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None)

    # Select good points
    good_prev_points = prev_points[status == 1]
    good_curr_points = curr_points[status == 1]

    # Check if we have enough good points
    if len(good_prev_points) < 4 or len(good_curr_points) < 4:
        print("Not enough good points for optical flow stabilization.")
        return curr_frame, curr_gray, prev_points

    # Compute transformation matrix
    matrix, _ = cv2.estimateAffinePartial2D(good_prev_points, good_curr_points)

    # Apply affine transformation
    height, width = curr_frame.shape[:2]
    stabilized_frame = cv2.warpAffine(curr_frame, matrix, (width, height))

    return stabilized_frame, curr_gray, curr_points

def plot_centerpoints(input_video_path, output_video_path):
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

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Error: Unable to read the first frame from {input_video_path}")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}")

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a format suitable for the detector
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect objects in the frame
        detection_result = detector.detect(mp_image)

        # Classify the color characteristics in the bounding boxes
        classified_detections = classify_white(frame_rgb, detection_result.detections)

        # Visualize the center points of the bounding boxes
        frame_copy = np.copy(frame_rgb)
        annotated_frame, center_points = visualize_centers(frame_copy, classified_detections)

        # Convert annotated frame back to BGR for processing
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Stabilize the frame using optical flow
        stabilized_frame, prev_gray, prev_points = optical_flow_stabilize(prev_frame, annotated_frame_bgr, prev_gray, prev_points)
        prev_frame = stabilized_frame

        # Write the stabilized frame to the output video
        out.write(stabilized_frame)

    print(f"Total frames processed: {frame_count}")

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = 'path/to/your/input/video.mp4'
    output_video_path = 'path/to/your/output/video_with_centers.mp4'
    plot_centerpoints(input_video_path, output_video_path)
