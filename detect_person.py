"""Person detection with YOLO26 (Ultralytics).

Works on a live webcam or a recorded video file.
Draws a bounding box around every person it finds.
"""
import argparse

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="YOLO26 person detection")
    p.add_argument("--source", default="0",
                   help="0 = webcam (or another camera index), or a path to a video file")
    p.add_argument("--model", default="yolo26s.pt",
                   help="YOLO26 weights (downloaded automatically on first run)")
    p.add_argument("--conf", type=float, default=0.5,
                   help="minimum confidence to keep a detection")
    p.add_argument("--save", default=None,
                   help="optional path to save the annotated video, e.g. output.mp4")
    p.add_argument("--no-display", action="store_true",
                   help="do not open a window (useful on a server)")
    return p.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))

    while True:
        ok, frame = cap.read()
        if not ok:
            break  # end of video, or webcam stopped

        # classes=[0] keeps only the "person" class from the COCO dataset
        results = model(frame, classes=[0], conf=args.conf, verbose=False)

        count = 0
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"person {confidence:.2f}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            count += 1

        cv2.putText(frame, f"People: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if writer:
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("YOLO26 person detection (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
