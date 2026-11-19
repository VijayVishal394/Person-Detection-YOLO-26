# Person Detection using YOLO26

Detects people in real time and draws a bounding box around each one, using the
YOLO26 small model (`yolo26s.pt`) from Ultralytics. Works on a **live webcam** and on **recorded videos**.

## Setup
```bash
pip install -r requirements.txt
```
The model weights download automatically the first time you run it.

## Run
```bash
# webcam
python detect_person.py --source 0

# recorded video
python detect_person.py --source my_video.mp4

# save the annotated result
python detect_person.py --source my_video.mp4 --save output.mp4
```
Press `q` to quit.

## How it works
1. Read one frame at a time with OpenCV.
2. Run YOLO26 on the frame, keeping only the `person` class (class 0 in COCO).
3. Draw a box and confidence score for each detection, and show the people count.

## Limitations
- It finds people but cannot tell *who* they are (see the face recognition project for that).
- Small or heavily occluded people may be missed.
- Speed depends on your hardware; a GPU is much faster than CPU.
