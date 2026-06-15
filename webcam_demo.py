import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


IMG_SIZE = (224, 224)
MODEL_PATH = Path("models/vegetable_classifier.keras")
CLASS_NAMES_PATH = Path("outputs/class_names.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-time webcam classification.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to trained .keras model.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index, usually 0.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum confidence to show a class.")
    parser.add_argument("--width", type=int, default=1280, help="Requested webcam width.")
    parser.add_argument("--height", type=int, default=720, help="Requested webcam height.")
    return parser.parse_args()


def load_class_names():
    if CLASS_NAMES_PATH.exists():
        return json.loads(CLASS_NAMES_PATH.read_text())

    data_train = Path("data/train")
    if data_train.exists():
        return sorted([path.name for path in data_train.iterdir() if path.is_dir()])

    raise FileNotFoundError("Could not find class names. Train first or keep outputs/class_names.json.")


def prepare_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    return np.expand_dims(resized.astype(np.float32), axis=0)


def draw_overlay(frame, label, confidence, fps):
    text = f"{label}: {confidence:.1%}"
    fps_text = f"FPS: {fps:.1f}"

    cv2.rectangle(frame, (20, 20), (520, 115), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (35, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, fps_text, (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press Q to quit", (20, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model)
    class_names = load_class_names()

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}. Try --camera 1.")

    fps = 0.0
    previous_time = time.time()

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        batch = prepare_frame(frame)
        probabilities = model.predict(batch, verbose=0)[0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])

        label = class_names[index] if confidence >= args.confidence else "Unknown"

        now = time.time()
        instant_fps = 1.0 / max(now - previous_time, 1e-6)
        fps = instant_fps if fps == 0.0 else (0.9 * fps + 0.1 * instant_fps)
        previous_time = now

        draw_overlay(frame, label, confidence, fps)
        cv2.imshow("Fruit and Vegetable Classification", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
