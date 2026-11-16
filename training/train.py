"""Train, evaluate, and export the four-class waste CNN.

Expected dataset layout:
    dataset/
      glass/*.jpg
      metal/*.jpg
      paper/*.jpg
      plastic/*.jpg

Run from the project directory:
    python training/train.py --data dataset
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


CLASSES = ["plastic", "paper", "glass", "metal"]
IMAGE_SIZE = (224, 224)
SEED = 42


def collect_files(data_dir: Path):
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_dir}\n"
            "Create it or pass --data with the path to the dataset root."
        )

    files = []
    labels = []
    for label, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            available = ", ".join(sorted(path.name for path in data_dir.iterdir() if path.is_dir()))
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}\n"
                f"Expected folders: {', '.join(CLASSES)}\n"
                f"Folders found: {available or '(none)'}"
            )
        class_files = sorted(
            path for path in class_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not class_files:
            raise FileNotFoundError(f"No supported images found in {class_dir}")
        files.extend(str(path) for path in class_files)
        labels.extend([label] * len(class_files))
    return np.array(files), np.array(labels)


def make_dataset(files, labels, training):
    paths = tf.constant(files)
    targets = tf.constant(labels, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, targets))

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(len(files), seed=SEED, reshuffle_each_iteration=True)
    return dataset.batch(32).prefetch(tf.data.AUTOTUNE)


def build_model(include_augmentation=True):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")
    layers = [tf.keras.Input(shape=(*IMAGE_SIZE, 3))]
    if include_augmentation:
        layers.append(augmentation)
    layers.extend([
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(len(CLASSES)),
    ])
    model = tf.keras.Sequential(layers, name="waste_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def save_confusion_matrix(matrix, output_path):
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(matrix, cmap="Greens")
    axis.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)),
             xticklabels=CLASSES, yticklabels=CLASSES,
             xlabel="Predicted label", ylabel="True label",
             title="Waste CNN confusion matrix")
    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("training/output"))
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(SEED)

    files, labels = collect_files(args.data)
    train_files, holdout_files, train_labels, holdout_labels = train_test_split(
        files, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    validation_files, test_files, validation_labels, test_labels = train_test_split(
        holdout_files, holdout_labels, test_size=0.50, random_state=SEED, stratify=holdout_labels
    )
    print(f"Images: {len(files)} | train: {len(train_files)} | "
          f"validation: {len(validation_files)} | test: {len(test_files)}")

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(output_dir / "best.keras", monitor="val_accuracy", save_best_only=True),
    ]
    model.fit(
        make_dataset(train_files, train_labels, True),
        validation_data=make_dataset(validation_files, validation_labels, False),
        epochs=args.epochs,
        callbacks=callbacks,
    )

    logits = model.predict(make_dataset(test_files, test_labels, False), verbose=0)
    predictions = np.argmax(logits, axis=1)
    report = classification_report(
        test_labels, predictions, target_names=CLASSES, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(test_labels, predictions, labels=range(len(CLASSES)))
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_confusion_matrix(matrix, output_dir / "confusion-matrix.png")
    print(classification_report(test_labels, predictions, target_names=CLASSES, zero_division=0))

    export_dir = output_dir / "browser-model"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    inference_model = build_model(include_augmentation=False)
    inference_model.set_weights(model.get_weights())
    inference_model.save(export_dir / "model.keras")
    inference_model.export(export_dir / "saved_model")
    print(f"Saved Keras model and metrics in {output_dir}")
    print("Convert the inference model with:")
    print(f"tensorflowjs_converter --input_format=tf_saved_model "
          f"--output_format=tfjs_graph_model {export_dir / 'saved_model'} web/model")


if __name__ == "__main__":
    main()
