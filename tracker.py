import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def main():
    # Initialize YOLOv8 model - using a pre-trained nano version for speed.
    # You can change to 'yolov8s.pt' or another version if desired.
    model = YOLO('yolov8n.pt')
    # Initialize the Deep SORT tracker.
    # max_age: Number of consecutive frames without a matching detection before a track is terminated.
    tracker = DeepSort(max_age=30)

    # Set up video capture. Option 1: from a webcam. Option 2: from a video file.
    cap = cv2.VideoCapture(0)  # For webcam feed
    # Uncomment the following line to use a video file:
    #cap = cv2.VideoCapture(r"path to video")

    if not cap.isOpened():
        print("Error opening video source")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame to read - exiting")
            break

        # Run YOLOv8 inference on the current frame.
        # The result is returned as a list containing detection results.
        results = model(frame, verbose=False)[0]

        # Prepare a list for detections that Deep SORT expects.
        detections = []
        # Extract bounding boxes, confidences, and class IDs from the YOLO output.
        boxes = results.boxes.xyxy.cpu().numpy()  # Format: [x1, y1, x2, y2]
        confidences = results.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results.boxes.cls.cpu().numpy()  # Class IDs

        # Loop over detections and filter by class, if needed.
        for box, conf, cls in zip(boxes, confidences, class_ids):
            # Here, we filter to track only "person". In the COCO dataset, person has class index 0.
            if int(cls) != 0:
                continue

            # Convert bounding box from [x1, y1, x2, y2] to [x, y, width, height]
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            detections.append(([x1, y1, width, height], float(conf), int(cls)))

        # Update tracker with current detections.
        # The update function returns a list of track objects.
        tracks = tracker.update_tracks(detections, frame=frame)

        # Loop through each valid track and draw the bounding box and track ID.
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id  # Unique identifier for the tracked object.
            bbox = track.to_ltrb()  # Get bounding box in [left, top, right, bottom] format.
            
            # Draw a rectangle around the object.
            cv2.rectangle(frame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        (0, 255, 0), 2)
            # Put the track ID text above the bounding box.
            cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the processed frame.
        cv2.imshow('YOLOv8 + Deep SORT Tracking', frame)

        # Press 'q' to exit the loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up: release the video capture object and close display windows.
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()




