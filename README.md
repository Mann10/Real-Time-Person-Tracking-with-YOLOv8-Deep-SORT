# Real-Time Person Tracking with YOLOv8 & Deep SORT

## Overview
This project implements real-time person tracking using **YOLOv8** for object detection and **Deep SORT** for tracking. The system assigns a unique ID to each detected person and follows them across video frames.

## Features
- Uses **YOLOv8 (nano version)** for fast person detection.
- Tracks individuals across frames using **Deep SORT**.
- Works with both **webcam** and **video files**.
- Displays bounding boxes and unique IDs on the tracked persons.
- Simple setup and easy to modify.

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/real-time-person-tracking.git
cd real-time-person-tracking
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Then, run:
```bash
pip install -r requirements.txt
```

#### Dependencies include:
- OpenCV (`cv2`)
- NumPy
- Ultralytics YOLO
- Deep SORT Realtime

## Usage
### Running with a Video File
Replace the video path in `cap = cv2.VideoCapture(<video_path>)` inside `main()` and run:
```bash
python tracking.py
```

### Running with a Webcam
Uncomment `cap = cv2.VideoCapture(0)` in `main()` and run:
```bash
python tracking.py
```

## How It Works
1. **YOLOv8 detects objects** in each frame and filters out only "persons".
2. **Deep SORT tracks individuals**, assigning unique IDs.
3. **Bounding boxes and IDs** are drawn on the frame.
4. **Video is displayed in real-time**, press **'q'** to exit.

## Example Output
![Tracking Example](https://via.placeholder.com/800x400.png?text=Tracking+Example)

## Future Enhancements
- Multi-camera tracking support.
- Export tracking data to a CSV file.
- Implement face recognition for re-identification.

## License
This project is open-source and licensed under the MIT License.

## Acknowledgments
- **Ultralytics YOLO**: For real-time object detection.
- **Deep SORT**: For multi-object tracking.
