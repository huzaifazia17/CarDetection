import cv2
import time
import sys

# Constant: Path to the Haar cascade file for object detection
CASCADE_PATH = 'cars.xml'  

# Load the Haar cascade classifier
def initialize_cascade(cascade_path):
    object_cascade = cv2.CascadeClassifier(cascade_path)
    if object_cascade.empty():
        raise IOError(f"Failed to load the cascade classifier from {cascade_path}")
    print("Cascade classifier initialized successfully.")
    return object_cascade

# Detect objects in a video frame
def detect_objects(frame, object_cascade):
    # Convert frame to grayscale for detection
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    detected_objects = object_cascade.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=5)
    return detected_objects

# Draw bounding boxes around detected objects
def annotate_objects(frame, objects):
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Process the video input frame by frame
def analyze_video(video_file, object_cascade):
    video_capture = cv2.VideoCapture(video_file)
    if not video_capture.isOpened():
        raise IOError(f"Unable to open video file {video_file}")

    # Retrieve video properties for logging
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(f"Video Details: {total_frames} frames, {video_fps:.2f} FPS")

    start_time = time.time()
    frame_counter = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Perform detection and annotation on the current frame
        objects = detect_objects(frame, object_cascade)
        annotate_objects(frame, objects)
        frame_counter += 1

        # Print status update every 10 frames
        if frame_counter % 10 == 0:
            print(f"Processed {frame_counter}/{total_frames} frames...")

    # Log total processing duration
    end_time = time.time()
    print(f"Total Processing Time: {end_time - start_time:.2f} seconds")
    print("Video analysis completed.")

    video_capture.release()

# Entry point for initializing the detection process
def main():
    if len(sys.argv) < 2:
        print("Usage: python object_detector.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    object_cascade = initialize_cascade(CASCADE_PATH)
    analyze_video(video_file, object_cascade)

if __name__ == "__main__":
    main()
