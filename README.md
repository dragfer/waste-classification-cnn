# CSN441 Mini Project Report

## Waste Classification Using Images

**Project Number:** 4  
**Course:** Fuzzy Logic and Neural Network (CSN441)  
**Submission Date:** [Enter submission date]

### Group Members

| Name | ERP ID |
|---|---|
| [Enter name] | [Enter ERP ID] |
| [Enter name] | [Enter ERP ID] |
| [Enter name] | [Enter ERP ID] |

> Replace every item enclosed in square brackets before submitting the report.

---

## Abstract

Waste materials are often mixed before collection, which makes recycling and
manual sorting more difficult. This project presents a convolutional neural
network (CNN) for classifying waste images into four material categories:
plastic, paper, glass, and metal.

The model was trained using images from the TrashNet dataset. Images were
resized, normalized, labelled, augmented, and divided into training,
validation, and test sets. A custom CNN was developed using TensorFlow and
Keras. The model was evaluated using accuracy, precision, recall, F1-score,
and a confusion matrix.

The reported model achieved an accuracy of 60.26% on 302 held-out test
images. The training program also generates a classification report and
confusion matrix for analysis.

**Keywords:** waste classification, convolutional neural network, CNN,
image classification, TensorFlow, Keras

---

## 1. Problem Definition

### 1.1 Background

Waste is generated in large quantities, but recyclable materials are often
placed in the same container. When paper, plastic, glass, and metal are mixed,
the recycling process becomes more difficult and the quality of recovered
material can decrease. Manual sorting can separate these materials, but it is
time-consuming, repetitive, and affected by human judgement.

Computer vision provides a way to assist this process. A trained image
classifier can learn visual patterns from labelled examples and predict the
material category of a new waste image.

### 1.2 Problem Statement

The aim of this project is to develop a CNN-based system that receives an
image of a waste item and classifies it into one of the following categories:

1. Plastic
2. Paper
3. Glass
4. Metal

The system assumes that the image contains one main waste object.

### 1.3 Objectives

The objectives of the project are:

1. To obtain a labelled image dataset for four waste categories.
2. To preprocess the images for CNN training.
3. To design and implement a suitable CNN architecture.
4. To train the model using TensorFlow and Keras.
5. To evaluate the model using appropriate classification metrics.
6. To analyse the model's strengths, errors, and limitations.
7. To generate reproducible evaluation outputs.
8. To demonstrate the trained classifier using test images.

### 1.4 Intended Use

The system is intended for academic demonstration and as a prototype for a
future waste-segregation assistant. A user selects an image, and the system
returns the predicted material together with the scores for all four classes.

The system is not a certified recycling decision system. Local recycling
rules should always be consulted before disposing of an item.

---

## 2. Dataset Collection

### 2.1 Dataset Source

The project uses the publicly available TrashNet dataset:

**Source:** https://github.com/garythung/trashnet

The original dataset contains approximately 2,527 labelled images across six
categories: glass, paper, cardboard, plastic, metal, and trash.

This project uses four categories required for the selected project scope.
The cardboard and trash categories were excluded.

### 2.2 Dataset Categories

| Category | Description |
|---|---|
| Plastic | Plastic bottles, containers, and polymer-based objects |
| Paper | Paper sheets, printed material, and paper packaging |
| Glass | Glass bottles, jars, and other glass objects |
| Metal | Cans and other metallic objects |

### 2.3 Dataset Distribution

The dataset used for the reported training run contains 2,008 images.

| Category | Number of images |
|---|---:|
| Plastic | 493 |
| Paper | 594 |
| Glass | 505 |
| Metal | 416 |
| **Total** | **2,008** |

The dataset is arranged in four directories:

```text
dataset/
├── glass/
├── metal/
├── paper/
└── plastic/
```

The categories are not perfectly balanced. Metal has the fewest images,
which is considered when interpreting the class-specific recall and F1-score.

**Figure 1: Dataset directory structure**  

**[PASTE FIGURE 1 IMAGE HERE — dataset folder screenshot]**

*Paste a screenshot showing the `plastic`, `paper`, `glass`, and `metal`
folders. See Appendix C.1 for the exact image requirement.*

---

## 3. Data Preprocessing

### 3.1 File Validation

The training program validates that all four expected category directories
are present. It supports JPG, JPEG, PNG, and WebP image files. If a category
directory is missing or contains no supported images, training stops with an
error rather than continuing with an incomplete dataset.

### 3.2 Image Resizing

The source images have different dimensions. Each image is converted to RGB
format and resized to 224 × 224 pixels. This creates a consistent input shape
for the CNN:

```text
224 × 224 × 3
```

The three channels represent red, green, and blue colour values.

### 3.3 Pixel Normalization

Original pixel values are integers between 0 and 255. They are converted to
floating-point values between 0 and 1 using:

```text
normalized pixel = pixel value / 255
```

Normalization helps maintain stable numerical values during training.

### 3.4 Label Encoding

The fixed label order is:

| Numeric label | Category |
|---:|---|
| 0 | Plastic |
| 1 | Paper |
| 2 | Glass |
| 3 | Metal |

This order is used consistently during training and evaluation.

### 3.5 Dataset Division

The complete dataset was divided using stratified sampling:

| Subset | Percentage | Number of images |
|---|---:|---:|
| Training | 70% | 1,404 |
| Validation | 15% | 302 |
| Test | 15% | 302 |
| **Total** | **100%** | **2,008** |

The training set is used to update model weights. The validation set is used
to monitor generalization and select the best training state. The test set is
kept separate until final evaluation.

The random seed was set to 42 so that the split can be reproduced.

### 3.6 Data Augmentation

The training images were augmented using:

- Random horizontal flips.
- Small random rotations.
- Small random zoom changes.

Augmentation increases the variety of training examples and helps reduce
overfitting. It is not applied to validation or test images. It is also
not applied to validation or test images, so the evaluation remains
deterministic.

---

## 4. Model Development

### 4.1 Selection of CNN

A CNN was selected because the input to the system is a still image. CNNs
are designed to learn spatial patterns such as edges, shapes, textures, and
object boundaries.

An RNN or LSTM was not required because the project does not process a
sequence of video frames. If the project were extended to video-based waste
classification, a CNN could extract features from each frame and an LSTM
could analyse the sequence.

### 4.2 CNN Architecture

The model is a compact sequential CNN:

| Layer | Configuration | Purpose |
|---|---|---|
| Input | 224 × 224 × 3 | Receives an RGB image |
| Conv2D | 32 filters, 3 × 3 kernel | Learns low-level features |
| MaxPooling2D | Default pooling | Reduces spatial dimensions |
| Conv2D | 64 filters, 3 × 3 kernel | Learns intermediate patterns |
| MaxPooling2D | Default pooling | Reduces feature-map size |
| Conv2D | 128 filters, 3 × 3 kernel | Learns deeper material features |
| MaxPooling2D | Default pooling | Compresses spatial information |
| GlobalAveragePooling2D | — | Produces a compact representation |
| Dropout | 0.35 | Reduces overfitting |
| Dense | 64 neurons, ReLU | Combines learned features |
| Dropout | 0.25 | Further regularization |
| Dense | 4 output neurons | Produces four class scores |

The convolutional layers learn increasingly complex visual representations.
Pooling reduces the computational size of the feature maps. Global average
pooling uses fewer parameters than a large fully connected layer. Dropout
reduces the chance of memorizing individual training images.

The final output contains four logits in this order:

```text
plastic, paper, glass, metal
```

**Figure 2: CNN architecture diagram**  

**[PASTE FIGURE 2 IMAGE HERE — CNN architecture diagram]**

*Paste a diagram showing the input, convolution, pooling, global average
pooling, dropout, dense, and output layers. See Appendix C.1.*

---

## 5. Model Training

### 5.1 Software Environment

| Component | Version or requirement |
|---|---|
| Programming language | Python 3.10–3.13 |
| Deep-learning framework | TensorFlow 2.21.0 |
| Neural-network API | Keras |
| Evaluation library | scikit-learn |
| Plotting library | Matplotlib |
| Output format | Keras model and evaluation artifacts |

Python 3.13 was used for the reported run. TensorFlow 2.21.0 is pinned in
the project requirements file.

### 5.2 Training Configuration

| Setting | Value |
|---|---|
| Batch size | 32 |
| Maximum epochs | 20 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Sparse categorical cross-entropy |
| Output mode | Four-class logits |
| Early stopping patience | 4 validation-loss regressions |
| Random seed | 42 |

### 5.3 Training Procedure

The model was trained using the training subset. After each epoch, the
validation subset was used to monitor performance. Early stopping restored
the best weights when validation loss stopped improving. This prevents a
later epoch with poorer validation performance from being selected.

The training command was:

```powershell
python training\train.py `
  --data C:\code\WasteClassification\dataset `
  --epochs 20
```

The training script generates:

```text
training/output/
├── best.keras
├── metrics.json
└── confusion-matrix.png
```

**Figure 3: Training output**  

**[PASTE FIGURE 3 IMAGE HERE — training command and epoch output]**

*Paste a readable terminal screenshot containing the training command, image
counts, and Keras epoch output. See Appendix C.1.*

---

## 6. Model Evaluation

### 6.1 Evaluation Method

The final evaluation was performed on the independent test set containing
302 images. These images were not used for updating model weights or making
early-stopping decisions.

The following measures were used:

- **Accuracy:** Overall proportion of correctly classified test images.
- **Precision:** Proportion of predictions for a class that were correct.
- **Recall:** Proportion of actual examples of a class that were detected.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Table showing correct predictions and class
  confusions.

### 6.2 Overall Results

| Metric | Result |
|---|---:|
| Accuracy | 60.26% |
| Macro precision | 65.03% |
| Macro recall | 57.85% |
| Macro F1-score | 58.16% |
| Weighted precision | 64.36% |
| Weighted recall | 60.26% |
| Weighted F1-score | 59.00% |

### 6.3 Class-Level Results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Plastic | 85.11% | 54.05% | 66.12% | 74 |
| Paper | 53.25% | 92.13% | 67.49% | 89 |
| Glass | 56.06% | 48.68% | 52.11% | 76 |
| Metal | 65.71% | 36.51% | 46.94% | 63 |

The results are stored in `training/output/metrics.json`.

**Figure 4: Classification report**  

**[PASTE FIGURE 4 IMAGE HERE — classification report]**

*Paste the terminal classification report or a screenshot of the generated
metrics table. See Appendix C.1.*

**Figure 5: Confusion matrix**  

**[PASTE FIGURE 5 IMAGE HERE — confusion matrix]**

*Paste the generated file `training/output/confusion-matrix.png`. See
Appendix C.1.*

---

## 7. Result Analysis

### 7.1 Strengths

Plastic achieved the highest precision at 85.11%. This means that when the
model predicts plastic, the prediction is often correct.

Paper achieved the highest recall at 92.13%. The model therefore identifies
most paper images in the test set, although its lower precision indicates
that some non-paper images are also predicted as paper.

The evaluation output displays separate precision, recall, and F1-score
values for all four classes rather than only the overall accuracy.

### 7.2 Errors

Metal achieved the lowest recall at 36.51% and the lowest F1-score at
46.94%. Glass also had a relatively low recall of 48.68%. These errors may
result from reflections, transparency, similar backgrounds, object shape,
lighting, and visual similarity between materials.

The class imbalance may also affect the results. The metal category has 416
images, compared with 594 paper images. The model consequently had fewer
metal examples from which to learn.

### 7.3 Limitations

The current system:

- Assumes that one main object is present in the image.
- Does not detect multiple objects or draw bounding boxes.
- May produce incorrect high-confidence predictions.
- Has not been trained on all possible waste products.
- Was evaluated on a relatively small and controlled dataset.
- Does not determine whether an item is recyclable under local regulations.

The test accuracy of 60.26% is suitable for an academic prototype but not
for unsupervised industrial sorting.

---

## 8. Conclusion

This project developed a complete CNN-based waste image classification
system for plastic, paper, glass, and metal. The work included dataset
collection, preprocessing, augmentation, CNN design, model training,
independent evaluation, metric generation, and result analysis.

The model achieved 60.26% accuracy on the held-out test set. It learned useful
visual patterns, particularly for identifying paper and making precise
plastic predictions. However, the lower recall for metal and glass shows
that the model requires additional data and refinement.

The objectives of the academic project were achieved. The resulting system
demonstrates how a CNN can be trained and deployed for image-based waste
classification while also showing the importance of evaluating each class
individually rather than relying only on overall accuracy.

---

## 9. Future Scope

### Dataset Improvements

1. Collect images from local environments, phones, and different cameras.
2. Increase the number of metal and glass examples.
3. Balance the number of images in each category.
4. Include varied backgrounds, lighting, orientation, and partial occlusion.

### Model Improvements

1. Compare the custom CNN with MobileNetV2 or another lightweight
   transfer-learning model.
2. Tune the image size, learning rate, dropout, and augmentation settings.
3. Use class weights if class imbalance continues to affect recall.
4. Add a validated uncertainty threshold.
5. Compare results across multiple train/test seeds.

### Application Improvements

1. Detect and classify multiple waste objects in one image.
2. Add object localization using an object-detection model.
3. Connect predictions to local recycling instructions.
4. Deploy the trained model in a mobile or desktop application.
5. Integrate the system with a camera-based smart-bin prototype.

---

## 10. Demonstration

### 10.1 Training Demonstration

Run the training program with the labelled dataset:

```powershell
python training\train.py `
  --data C:\code\WasteClassification\dataset `
  --epochs 20
```

The terminal output shows the image counts, training progress, validation
monitoring, and final classification report. The generated confusion matrix
and metrics file provide the output required for result analysis.

### 10.2 Demonstration Procedure

1. Confirm that the dataset contains the four expected class folders.
2. Run the training command from Appendix A.
3. Record the training, validation, and test image counts.
4. Record the final classification report.
5. Open `training/output/confusion-matrix.png`.
6. Compare the precision, recall, and F1-score for each category.
7. Use a held-out test image to explain how the trained classifier produces
   a class prediction.

### 10.3 Example Output

The demonstration should show:

```text
Images: 2008 | train: 1404 | validation: 302 | test: 302
```

It should also show the classification report containing accuracy,
precision, recall, F1-score, and support for plastic, paper, glass, and metal.

**Figure 6: Training and evaluation output**

**[PASTE FIGURE 6 IMAGE HERE — training command and classification report]**

*Paste a screenshot containing the training command, dataset counts, and
classification report.*

---

## Appendix A — Reproducibility Instructions

### A.1 Environment Setup

```powershell
cd C:\code\HacksagonBackend\waste-classification
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r training\requirements.txt
```

### A.2 Training and Evaluation

```powershell
python training\train.py `
  --data C:\code\WasteClassification\dataset `
  --epochs 20
```

### A.3 Project Files

| File | Purpose |
|---|---|
| `training/train.py` | Dataset preparation, CNN training, and evaluation |
| `training/requirements.txt` | Python training dependencies |

---

## Submission Checklist

- [ ] Enter all group member names and ERP IDs.
- [ ] Confirm the project title and project number.
- [ ] Insert the dataset screenshot.
- [ ] Insert the CNN architecture diagram.
- [ ] Insert the training-output screenshot.
- [ ] Insert the classification report.
- [ ] Insert the confusion matrix.
- [ ] Check that all reported numbers match the generated artifacts.
- [ ] Remove local paths if they should not appear in the final submission.
- [ ] Export the completed Word document as PDF.

---

## Appendix B — Complete Source Code

This appendix contains the complete source code used by the project. The code
is included so that the report documents the implementation, not only the
experimental results. The source files remain available in the project
directory and should be used when reproducing the experiment.

### B.1 Python Training and Evaluation Program

**File:** `training/train.py`

```python
"""Train, evaluate, and export the four-class waste CNN."""

import argparse
import json
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
            available = ", ".join(
                sorted(path.name for path in data_dir.iterdir() if path.is_dir())
            )
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}\n"
                f"Expected folders: {', '.join(CLASSES)}\n"
                f"Folders found: {available or '(none)'}"
            )

        class_files = sorted(
            path
            for path in class_dir.rglob("*")
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
        image = tf.image.decode_image(
            image, channels=3, expand_animations=False
        )
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(
            len(files), seed=SEED, reshuffle_each_iteration=True
        )
    return dataset.batch(32).prefetch(tf.data.AUTOTUNE)


def build_model(include_augmentation=True):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    layers = [tf.keras.Input(shape=(*IMAGE_SIZE, 3))]
    if include_augmentation:
        layers.append(augmentation)

    layers.extend(
        [
            tf.keras.layers.Conv2D(
                32, 3, activation="relu", padding="same"
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(
                64, 3, activation="relu", padding="same"
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(
                128, 3, activation="relu", padding="same"
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(len(CLASSES)),
        ]
    )

    model = tf.keras.Sequential(layers, name="waste_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        metrics=["accuracy"],
    )
    return model


def save_confusion_matrix(matrix, output_path):
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(matrix, cmap="Greens")
    axis.set(
        xticks=range(len(CLASSES)),
        yticks=range(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted label",
        ylabel="True label",
        title="Waste CNN confusion matrix",
    )

    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            axis.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("training/output")
    )
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(SEED)

    files, labels = collect_files(args.data)
    train_files, holdout_files, train_labels, holdout_labels = (
        train_test_split(
            files,
            labels,
            test_size=0.30,
            random_state=SEED,
            stratify=labels,
        )
    )
    validation_files, test_files, validation_labels, test_labels = (
        train_test_split(
            holdout_files,
            holdout_labels,
            test_size=0.50,
            random_state=SEED,
            stratify=holdout_labels,
        )
    )

    print(
        f"Images: {len(files)} | train: {len(train_files)} | "
        f"validation: {len(validation_files)} | test: {len(test_files)}"
    )

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    model.fit(
        make_dataset(train_files, train_labels, True),
        validation_data=make_dataset(
            validation_files, validation_labels, False
        ),
        epochs=args.epochs,
        callbacks=callbacks,
    )

    logits = model.predict(
        make_dataset(test_files, test_labels, False),
        verbose=0,
    )
    predictions = np.argmax(logits, axis=1)
    report = classification_report(
        test_labels,
        predictions,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        test_labels,
        predictions,
        labels=range(len(CLASSES)),
    )

    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    save_confusion_matrix(
        matrix,
        output_dir / "confusion-matrix.png",
    )
    print(
        classification_report(
            test_labels,
            predictions,
            target_names=CLASSES,
            zero_division=0,
        )
    )

    print(f"Saved Keras model and metrics in {output_dir}")


if __name__ == "__main__":
    main()
```

### B.2 Dependency Configuration

**File:** `training/requirements.txt`

```text
tensorflow==2.21.0
scikit-learn>=1.4,<2
matplotlib>=3.8,<4
```

### B.3 Training Output Contract

The training program produces the following artifacts:

| Artifact | Purpose |
|---|---|
| `training/output/best.keras` | Best CNN checkpoint selected using validation accuracy |
| `training/output/metrics.json` | Precision, recall, F1-score, and support for the test set |
| `training/output/confusion-matrix.png` | Visual representation of test-set predictions |

The model output contains four logits in the fixed class order `plastic` `paper`
`glass` `metal`. The logits are used to select the predicted class during
Python evaluation. The evaluation artifacts are generated directly by
`training/train.py` and are the authoritative results for this report.

## Appendix C ? Image Placement and Figure Classification

The following table identifies which image should be pasted at each location
in the report. The figure list focuses on the Python CNN training and
evaluation implementation.

| Figure | Paste this image | Source or action | Report location |
|---:|---|---|---|
| 1 | Dataset folder structure | Screenshot showing `plastic`, `paper`, `glass`, and `metal` folders | Section 2.3, after the dataset distribution table |
| 2 | CNN architecture diagram | Diagram based on the architecture table in Section 4.2 | Section 4.2, after the architecture explanation |
| 3 | Training execution | Screenshot of the PowerShell training command and epoch output | Section 5.3, after the training command |
| 4 | Classification report | Screenshot of the terminal classification report or metrics table | Section 6.3, after the class-level results |
| 5 | Confusion matrix | Use `training/output/confusion-matrix.png` | Section 6.3, after Figure 4 |
| 6 | Training and evaluation output | Screenshot containing dataset counts and final evaluation output | Section 10.3, after the demonstration procedure |

### C.1 Required Figure Blocks

```text
Figure 1: Dataset directory structure

[PASTE IMAGE HERE]

Source: Screenshot of the four class folders in the dataset directory.
```

```text
Figure 2: CNN architecture diagram

[PASTE IMAGE HERE]

Source: Diagram showing the input, convolution, pooling, global average
pooling, dropout, dense, and output layers.
```

```text
Figure 3: Training execution and epoch output

[PASTE IMAGE HERE]

Source: Screenshot of the training command, dataset counts, and Keras epoch
output.
```

```text
Figure 4: Test-set classification report

[PASTE IMAGE HERE]

Source: Screenshot of the precision, recall, F1-score, and support values.
```

```text
Figure 5: Confusion matrix

[PASTE IMAGE HERE]

Source: `training/output/confusion-matrix.png`.
```

```text
Figure 6: Training and evaluation output

[PASTE IMAGE HERE]

Source: Screenshot containing the training command, dataset counts, and
classification report.
```

### C.2 Image Formatting Instructions

1. Insert each image immediately below its matching figure heading.
2. Centre screenshots and diagrams on the page.
3. Crop terminal screenshots so that the command and relevant output remain
   readable.
4. Keep the confusion-matrix labels and numeric cell values visible.
5. Add one or two sentences below each image explaining what it demonstrates.
6. If an image is unavailable, leave the `[PASTE IMAGE HERE]` block in place
   until the final document review.
