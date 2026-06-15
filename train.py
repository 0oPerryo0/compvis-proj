import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


IMG_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 16
MODEL_PATH = Path("models/vegetable_classifier.keras")
OUTPUT_DIR = Path("outputs")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a vegetable image classifier.")
    parser.add_argument("--data-dir", default="data", help="Dataset root with train/validation/test folders.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--fine-tune-epochs", type=int, default=5, help="Extra fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Images per training batch.")
    parser.add_argument("--resume", action="store_true", help="Continue training from models/vegetable_classifier.keras.")
    return parser.parse_args()


def configure_gpu_memory():
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def load_dataset(path, shuffle, batch_size):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def prepare_dataset(dataset, training=False):
    if training:
        dataset = dataset.shuffle(1000)
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(num_classes):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="vegetable_mobilenetv2")
    compile_model(model, learning_rate=0.001)
    return model, base_model


def get_num_outputs(model):
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    return output_shape[-1]


def find_mobilenet_base(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            return layer
    return None


def check_resume_matches_dataset(model, class_names):
    saved_class_names_path = OUTPUT_DIR / "class_names.json"
    if saved_class_names_path.exists():
        saved_class_names = json.loads(saved_class_names_path.read_text())
        if saved_class_names != class_names:
            raise ValueError(
                "Saved class names do not match the current data/train folders. "
                "Use fresh training without --resume after adding, removing, or renaming classes."
            )

    model_outputs = get_num_outputs(model)
    if model_outputs != len(class_names):
        raise ValueError(
            f"Saved model has {model_outputs} output classes, but the dataset has {len(class_names)}. "
            "Use fresh training without --resume when the number of classes changes."
        )


def load_or_build_model(args, class_names):
    if args.resume:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Cannot resume because {MODEL_PATH} does not exist.")

        print(f"Resuming training from: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        check_resume_matches_dataset(model, class_names)
        compile_model(model, learning_rate=0.0001)
        return model, find_mobilenet_base(model)

    print("Starting fresh training from ImageNet MobileNetV2 weights.")
    return build_model(num_classes=len(class_names))


def plot_history(history, output_path):
    metrics = history.history
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(metrics["accuracy"], label="train")
    plt.plot(metrics["val_accuracy"], label="validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics["loss"], label="train")
    plt.plot(metrics["val_loss"], label="validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def evaluate_model(model, dataset, class_names):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(probabilities, axis=1))

    print("\nClassification report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    (OUTPUT_DIR / "classification_report.txt").write_text(report)

    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    display.plot(ax=ax, xticks_rotation=45, cmap="Blues")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_demo_predictions(model, dataset, class_names):
    for images, labels in dataset.take(1):
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        count = min(9, len(images))

        plt.figure(figsize=(9, 9))
        for index in range(count):
            plt.subplot(3, 3, index + 1)
            plt.imshow(images[index].numpy().astype("uint8"))
            predicted_name = class_names[predictions[index]]
            true_name = class_names[int(labels[index])]
            confidence = probabilities[index][predictions[index]]
            color = "green" if predicted_name == true_name else "red"
            plt.title(f"{predicted_name} ({confidence:.0%})\ntrue: {true_name}", color=color)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "demo_predictions.png", dpi=160)
        plt.close()
        break


def main():
    configure_gpu_memory()

    args = parse_args()
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not validation_dir.exists():
        raise FileNotFoundError(
            "Expected data/train and data/validation folders. See README.md for the folder layout."
        )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_ds = load_dataset(train_dir, shuffle=True, batch_size=args.batch_size)
    validation_ds = load_dataset(validation_dir, shuffle=False, batch_size=args.batch_size)
    class_names = train_ds.class_names
    (OUTPUT_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2))

    train_ds = prepare_dataset(train_ds, training=True)
    validation_ds = prepare_dataset(validation_ds)

    model, base_model = load_or_build_model(args, class_names)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    print(f"Training classes: {class_names}")
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    plot_history(history, OUTPUT_DIR / "training_history.png")

    if args.fine_tune_epochs > 0:
        if base_model is None:
            raise RuntimeError("Could not find the MobileNetV2 base model for fine-tuning.")

        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        compile_model(model, learning_rate=0.00005)
        model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
        )

    final_model = tf.keras.models.load_model(MODEL_PATH)
    eval_ds = load_dataset(test_dir, shuffle=False, batch_size=args.batch_size) if test_dir.exists() else validation_ds
    loss, accuracy = final_model.evaluate(eval_ds)
    print(f"\nFinal accuracy: {accuracy:.3f}")
    evaluate_model(final_model, eval_ds, class_names)
    save_demo_predictions(final_model, eval_ds, class_names)
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
