#!/usr/bin/env python3
"""
Video face swap: replaces the main character's face in inputs/source.mp4
with the face from inputs/face.jpg, keeping motion, lighting and audio.

Usage:
  python face_swap.py --source inputs/source.mp4 --face inputs/face.jpg --out output/result.mp4

Requires: insightface, onnxruntime, opencv-python-headless, numpy, ffmpeg (CLI).
The swapper model (inswapper_128.onnx) is downloaded by the workflow into models/.
"""
import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


def log(msg):
    print(f"[face-swap] {msg}", flush=True)


def load_models(model_path, det_size):
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    swapper = insightface.model_zoo.get_model(model_path, providers=["CPUExecutionProvider"])
    return app, swapper


def pick_source_face(app, face_img):
    faces = app.get(face_img)
    if not faces:
        sys.exit("No face found in the reference photo.")
    # largest face in the photo
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def pick_target_face(faces, prev_center, frame_w, frame_h):
    """Choose which detected face is the hero: prefer the one nearest to the
    previous hero position, otherwise the largest face in frame."""
    if not faces:
        return None
    if prev_center is not None:
        def dist(f):
            cx = (f.bbox[0] + f.bbox[2]) / 2
            cy = (f.bbox[1] + f.bbox[3]) / 2
            return ((cx - prev_center[0]) ** 2 + (cy - prev_center[1]) ** 2) ** 0.5
        nearest = min(faces, key=dist)
        # accept if reasonably close (within 25% of frame diagonal)
        if dist(nearest) < 0.25 * (frame_w ** 2 + frame_h ** 2) ** 0.5:
            return nearest
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--face", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="models/inswapper_128.onnx")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--all-faces", action="store_true",
                    help="swap every face in frame instead of only the hero")
    ap.add_argument("--min-face", type=int, default=24,
                    help="ignore faces smaller than this many pixels wide")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"Swapper model not found at {args.model}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp_video = args.out + ".noaudio.mp4"

    log("loading models")
    app, swapper = load_models(args.model, args.det_size)

    face_img = cv2.imread(args.face)
    if face_img is None:
        sys.exit("Could not read reference photo.")
    src_face = pick_source_face(app, face_img)
    log("reference face locked")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit("Could not open source video.")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"source {w}x{h} @ {fps:.2f}fps, {total} frames")

    writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    prev_center = None
    swapped = 0
    t0 = time.time()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = [f for f in app.get(frame) if (f.bbox[2] - f.bbox[0]) >= args.min_face]
        if faces:
            targets = faces if args.all_faces else [pick_target_face(faces, prev_center, w, h)]
            for tf in targets:
                frame = swapper.get(frame, tf, src_face, paste_back=True)
            hero = targets[0]
            prev_center = ((hero.bbox[0] + hero.bbox[2]) / 2, (hero.bbox[1] + hero.bbox[3]) / 2)
            swapped += 1
        writer.write(frame)
        i += 1
        if i % 50 == 0:
            el = time.time() - t0
            log(f"{i}/{total} frames ({swapped} swapped) {el:.0f}s elapsed")
    cap.release()
    writer.release()
    log(f"done: {i} frames, {swapped} with a swap")

    # Re-encode to H.264 and mux the original audio back in.
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", tmp_video, "-i", args.source,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        args.out,
    ]
    subprocess.run(cmd, check=True)
    os.remove(tmp_video)
    log(f"written {args.out}")


if __name__ == "__main__":
    main()
