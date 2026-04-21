import cv2
import numpy as np
import os


def dense_optical_flow(method, video_path, params=[], to_gray=False, output_path="output.mp4"):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}")
        return

    ret, old_frame = cap.read()
    if not ret or old_frame is None:
        print("Error: could not read first frame")
        cap.release()
        return

    h, w = old_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, new_frame = cap.read()
        if not ret or new_frame is None:
            break

        frame_copy = new_frame.copy()

        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        flow = method(old_frame, new_frame, None, *params)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        combined = np.hstack((frame_copy, bgr))
        out.write(combined)

        old_frame = new_frame

    cap.release()
    out.release()
    print(f"Saved output to {output_path}")