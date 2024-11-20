import cv2
import time
import sys

# Constants:
# Path to Haar cascade file for object detection
CASCADE_PATH = 'cars.xml'  
# Title for the video display window
DISPLAY_TITLE = 'Vehicle Detection'  

def initialize_cascade(cascade_path):
    """Initialize the Haar cascade for object detection."""
    cascade_classifier = cv2.CascadeClassifier(cascade_path)
    if cascade_classifier.empty():
        raise IOError(f"Failed to load the cascade classifier from {cascade_path}")
    return cascade_classifier

def detect_objects(frame, cascade_classifier):
    """Detect objects in a given video frame."""
    
    # Convert the frame to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    return cascade_classifier.detectMultiScale(grayscale_frame, scaleFactor=1.1, minNeighbors=5)

def annotate_frame(frame, detections):
    """Annotate the frame with rectangles around detected objects."""
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def handle_video_stream(video_file, cascade_classifier):
    """Process the video file and show detection results."""
    video_capture = cv2.VideoCapture(video_file)
    if not video_capture.isOpened():
        raise IOError(f"Unable to open video file {video_file}")

    # Retrieve basic video information
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(f"Video Information: {frame_count} frames, {video_fps} FPS")

    processing_start = time.time()
    processed_frames = 0

    while True:
        # Retrieve the next frame
        is_frame_read, current_frame = video_capture.read()  
        if not is_frame_read:
            break

        processed_frames += 1
        frame_start = time.time()

        # Perform object detection and annotate the frame
        detected_objects = detect_objects(current_frame, cascade_classifier)
        annotate_frame(current_frame, detected_objects)

        # Display frame details and detection results
        cv2.putText(current_frame, f'Frame: {processed_frames}/{frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(current_frame, f'Objects Detected: {len(detected_objects)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(DISPLAY_TITLE, current_frame)

        # Exit display on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Log frame processing duration
        frame_end = time.time()
        frame_duration = frame_end - frame_start
        print(f"Frame {processed_frames}: {frame_duration:.2f} seconds")

    # Log overall processing time
    processing_end = time.time()
    total_duration = processing_end - processing_start
    print(f"Total Processing Time: {total_duration:.2f} seconds")

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    """Entry point for initiating object detection."""
    if len(sys.argv) < 2:
        print("Usage: python vehicle_detection.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    cascade_classifier = initialize_cascade(CASCADE_PATH)
    handle_video_stream(video_file, cascade_classifier)

if __name__ == "__main__":
    main()
