import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


IMG_SIZE = (224, 224)
MODEL_PATH = Path("models/vegetable_classifier.keras")
CLASS_NAMES_PATH = Path("outputs/class_names.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the vegetable in one image.")
    parser.add_argument("--image", required=True, help="Path to an image file.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to trained .keras model.")
    return parser.parse_args()


def load_class_names():
    if CLASS_NAMES_PATH.exists():
        return json.loads(CLASS_NAMES_PATH.read_text())
    data_train = Path("data/train")
    if data_train.exists():
        return sorted([path.name for path in data_train.iterdir() if path.is_dir()])
    raise FileNotFoundError("Could not find class names. Keep data/train folders or outputs/class_names.json.")


def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model)
    class_names = load_class_names()

    image = tf.keras.utils.load_img(args.image, target_size=IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    batch = np.expand_dims(array, axis=0)

    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])

    print(f"Prediction: {class_names[index]}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
