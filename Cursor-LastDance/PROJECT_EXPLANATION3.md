# Dental X-ray Cavity Detection - Full Project Explanation

This document explains the whole project in a discussion-friendly way:

- what the project does
- why we chose this workflow
- how the dataset is used
- how the code is implemented
- what every important function does
- how the workspace is organized
- what to say when discussing it with the TA

Main implementation file:

```text
dental_xray_detection.py
```

Notebook runner:

```text
dental_xray_detection.ipynb
```

Portable dataset path inside the project:

```text
dataset/
```

Important: this project does not generate synthetic data. It uses the real TUFTS dental X-ray dataset.

---

## 1. Project Idea

The goal is to build a simple deep learning project that classifies dental X-ray images into two classes:

```text
0 = no_cavity
1 = cavity
```

The model receives a dental X-ray image and predicts whether the image has an annotated cavity/anomaly or not.

We use a CNN-based transfer learning approach with MobileNetV2. Instead of training a CNN from zero, we use a pretrained CNN backbone and train a small classifier head on our dental dataset.

---

## 2. Why This Project Fits The Neural Networks Course

This project connects directly to the college labs:

| Lab | Concept | How It Appears In This Project |
|---|---|---|
| Lab 06 | Keras Sequential API | The model is built with `tf.keras.Sequential` |
| Lab 06 | `compile`, `fit`, `evaluate` | Used for training and testing |
| Lab 07 | CNN image classification | MobileNetV2 is a CNN feature extractor |
| Lab 07 | Data augmentation | Random flip, small rotation, small zoom |
| Lab 07 | Dropout | Reduces overfitting in the dense head |
| Lab 08 | Transfer learning | MobileNetV2 pretrained backbone |
| Lab 08 | Freezing and fine-tuning | Phase 1 freezes backbone, Phase 2 unfreezes last 10 layers |

The implementation is intentionally simple so it is easy to explain in a discussion.

---

## 3. Workspace Architecture

The project workspace is:

```text
D:\8th term\NN\Codex-LastDance
```

Current project files:

```text
Codex-LastDance/
  dental_xray_detection.py       main Python implementation
  dental_xray_detection.ipynb    notebook version that calls the Python functions
  README.md                      quick start instructions
  PROJECT_EXPLANATION.md         this detailed explanation
  requirements.txt               required Python libraries
  models/                        created after training
  outputs/                       created after training/evaluation
```

After training, the code saves:

```text
models/
  best_model.keras

outputs/
  augmented_samples.png
  phase1_history.png
  phase2_history.png
  training_history.png
  confusion_matrix.png
  predictions.png
```

### What Each File Does

| File | Purpose |
|---|---|
| `dental_xray_detection.py` | Full project code: load data, build model, train, fine-tune, evaluate |
| `dental_xray_detection.ipynb` | Notebook interface for presenting/running the same code step by step |
| `README.md` | Short instructions for installation and running |
| `PROJECT_EXPLANATION.md` | Detailed project explanation for college discussion |
| `requirements.txt` | Python packages needed for the project |

---

## 4. Dataset Architecture

Your real dataset is copied inside the workspace so the project can be moved to another laptop:

```text
Codex-LastDance/dataset
```

The detected structure is:

```text
dataset/
  TUFTS/
    Radiographs/
      training_images/
      testing_images/
    bboxes/
      trainBoundryBoxes.csv
      testBoundryBoxes.csv
```

The image folders contain panoramic dental X-ray images:

```text
training_images/   980 images
testing_images/     20 images
```

The CSV files contain bounding-box annotations:

```text
trainBoundryBoxes.csv
testBoundryBoxes.csv
```

Example CSV structure:

```text
imageID,class,x-min,y-min,width,height
149.JPG,4,651,623,83,61
727.JPG,2,1081,592,50,59
```

For this simple classification project, we do not train an object detector. We only use the CSV to create binary image labels.

---

## 5. Labeling Rule

The dataset has images and bounding boxes, but the project is image classification, not object detection.

So we use this simple rule:

```text
If an image appears in the bbox CSV -> label = cavity
If an image does not appear in the bbox CSV -> label = no_cavity
```

In code:

```python
def read_positive_image_ids(csv_path: Path) -> set[str]:
    positive_ids = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            image_id = row.get("imageID", "").strip()
            if image_id:
                positive_ids.add(image_id)
    return positive_ids
```

This function reads the CSV and collects all image filenames that have annotations.

Then labels are created here:

```python
labels = np.array([1 if path.name in positive_ids else 0 for path in image_paths], dtype=np.int32)
```

Meaning:

```text
1 = cavity
0 = no_cavity
```

Detected dataset statistics:

```text
Train images: 980 total | 335 cavity | 645 no_cavity
Test images:   20 total |   4 cavity |  16 no_cavity
```

Because the classes are imbalanced, the code uses class weights.

---

## 6. Why We Use Class Weights

The training set has more no-cavity images than cavity images:

```text
no_cavity = 645
cavity    = 335
```

If we train normally, the model may become biased toward the majority class.

Class weights give more importance to the smaller class.

Code:

```python
def balanced_class_weights(labels: np.ndarray) -> dict[int, float] | None:
    counts = np.bincount(labels.astype(int), minlength=2)
    if np.any(counts == 0):
        return None

    total = float(len(labels))
    return {
        0: total / (2.0 * float(counts[0])),
        1: total / (2.0 * float(counts[1])),
    }
```

For your dataset, the weights are approximately:

```text
0 = no_cavity -> 0.7597
1 = cavity    -> 1.4627
```

So the cavity class receives a larger weight during training.

---

## 7. Full Workflow Sequence

The whole project runs in this order:

```text
1. Read command-line settings
2. Set random seeds for reproducibility
3. Check the real dataset exists
4. Detect TUFTS dataset layout
5. Read bbox CSV files
6. Convert image filenames into binary labels
7. Build TensorFlow datasets
8. Resize images to 96x96
9. Normalize pixel values to 0..1
10. Cache and prefetch data for CPU speed
11. Build light data augmentation
12. Build MobileNetV2 transfer-learning model
13. Phase 1: freeze MobileNetV2, train classifier head
14. Phase 2: unfreeze last 10 MobileNetV2 layers, fine-tune
15. Load best saved model
16. Evaluate on test dataset
17. Save plots and prediction examples
18. Print final result summary
```

As a diagram:

```text
Real TUFTS Images + bbox CSV
              |
              v
    Binary labels: no_cavity/cavity
              |
              v
       TensorFlow Dataset
              |
              v
 Resize 96x96 + normalize 0..1
              |
              v
     Light data augmentation
              |
              v
      MobileNetV2 CNN backbone
              |
              v
   GlobalAveragePooling2D
              |
              v
 Dense(64) + Dropout(0.3)
              |
              v
 Dense(1, sigmoid)
              |
              v
 Binary prediction
```

---

## 8. Important Constants

At the top of the code:

```python
IMG_SIZE = (96, 96)
BATCH_SIZE = 16
EPOCHS_HEAD = 5
EPOCHS_FINETUNE = 3
SEED = 42

DATASET_PATH = Path("dataset")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
BEST_MODEL_PATH = MODEL_DIR / "best_model.keras"
```

### Why `96x96`?

The project is CPU-friendly. Smaller images train much faster than `224x224`.

`96x96` is a good compromise:

- much faster on CPU
- still enough for a project demo
- accepted by MobileNetV2

### Why batch size `16`?

Batch size `16` keeps memory usage low and is stable on CPU.

### Why seed `42`?

The seed makes shuffling and initialization more reproducible.

---

## 9. Dataset Detection

The code first checks if the dataset is TUFTS style:

```python
def find_tufts_root(dataset_dir: Path) -> Path | None:
    for candidate in (dataset_dir, dataset_dir / "TUFTS"):
        has_images = (candidate / "Radiographs" / "training_images").exists()
        has_test_images = (candidate / "Radiographs" / "testing_images").exists()
        has_train_csv = (candidate / "bboxes" / "trainBoundryBoxes.csv").exists()
        has_test_csv = (candidate / "bboxes" / "testBoundryBoxes.csv").exists()
        if has_images and has_test_images and has_train_csv and has_test_csv:
            return candidate
    return None
```

This is useful because the code works when the user passes either:

```text
dataset
```

or:

```text
dataset\TUFTS
```

Both work.

---

## 10. Image Loading Pipeline

The images are loaded using TensorFlow.

Code:

```python
def load_image_from_path(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.reshape(tf.cast(label, tf.float32), (1,))
    return image, label
```

What happens here:

1. Read image file from disk.
2. Decode it as an RGB image with 3 channels.
3. Resize it to `96x96`.
4. Convert pixels to float.
5. Normalize from `0..255` to `0..1`.
6. Return image and binary label.

The dataset is then batched, cached, and prefetched:

```python
return dataset.map(load_image_from_path, num_parallel_calls=autotune).batch(batch_size).cache().prefetch(autotune)
```

### Why cache and prefetch?

On CPU, training can be slowed down by file loading. `cache()` keeps processed batches available, and `prefetch()` prepares the next batch while the model is training on the current batch.

---

## 11. Data Augmentation

Data augmentation creates slightly modified versions of training images so the model generalizes better.

Code:

```python
def build_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="light_augmentation",
    )
```

We keep augmentation light because these are medical images. Heavy transformations can distort important medical patterns.

Used augmentations:

| Layer | Purpose |
|---|---|
| `RandomFlip("horizontal")` | Helps model handle left/right variation |
| `RandomRotation(0.05)` | Small rotation only |
| `RandomZoom(0.1)` | Small zoom variation |

The code also saves a preview:

```text
outputs/augmented_samples.png
```

Augmentation is applied outside the model and only to the training dataset:

```python
augmented_train_dataset = apply_training_augmentation(train_dataset, data_augmentation)
```

This keeps validation, testing, and prediction stable. The saved model itself does not randomly modify test images.

---

## 12. Why Transfer Learning?

Training a CNN from scratch requires a large dataset and strong hardware. Our dataset is small, especially the test set.

Transfer learning solves this by using a CNN that already learned useful visual features from a large dataset.

We use:

```text
MobileNetV2
```

### Why MobileNetV2?

MobileNetV2 is:

- lightweight
- faster than many large CNNs
- suitable for CPU
- available in Keras
- pretrained on ImageNet

Even though ImageNet is not medical data, early CNN layers learn general features such as edges, curves, textures, and shapes. These are still useful for X-ray images.

---

## 13. Model Architecture

The model is built in `build_model`.

Code:

```python
dental_model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(*IMG_SIZE, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ],
    name="dental_model",
)
```

Layer-by-layer:

| Layer | What It Does |
|---|---|
| `Input(96, 96, 3)` | Receives RGB X-ray image |
| `MobileNetV2` | Extracts visual/CNN features |
| `GlobalAveragePooling2D` | Converts feature maps into a compact feature vector |
| `Dense(64, relu)` | Learns task-specific patterns |
| `Dropout(0.3)` | Reduces overfitting |
| `Dense(1, sigmoid)` | Outputs probability of cavity |

### Why sigmoid?

The task is binary classification:

```text
no_cavity vs cavity
```

So we use one output neuron:

```python
tf.keras.layers.Dense(1, activation="sigmoid")
```

The output is a probability between 0 and 1:

```text
>= 0.5 -> cavity
<  0.5 -> no_cavity
```

---

## 14. Model Compilation

Phase 1 compilation:

```python
dental_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
```

### Why Adam?

Adam is a common optimizer that works well for many deep learning tasks. It adapts the learning rate internally for each parameter.

### Why binary crossentropy?

Because this is binary classification.

For binary classification:

```text
output layer = sigmoid
loss         = binary_crossentropy
```

---

## 15. Training Strategy

The training has two phases.

### Phase 1 - Train Head Only

First, MobileNetV2 is frozen:

```python
base_model.trainable = False
```

Only the new dense classifier head is trained.

Code:

```python
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weight,
)
```

Why this phase is useful:

- fast
- prevents destroying pretrained features
- trains only the new classifier layers

### Phase 2 - Fine-Tune Last 10 Layers

After the head learns something, we unfreeze only the last 10 layers of MobileNetV2:

```python
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False
```

Then we recompile with a much smaller learning rate:

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
```

Why low learning rate?

Fine-tuning changes pretrained weights. If the learning rate is too high, the model can destroy useful pretrained features. A low learning rate makes small careful updates.

---

## 16. Callbacks

Callbacks help training stop at the best point and save the best model.

Code:

```python
return [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),
]
```

### ModelCheckpoint

Saves the best model based on validation accuracy:

```text
models/best_model.keras
```

### EarlyStopping

Stops training if validation loss stops improving. This saves CPU time and helps avoid overfitting.

---

## 17. Evaluation

After training, the best saved model is loaded:

```python
best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
```

Then it is evaluated:

```python
loss, accuracy = best_model.evaluate(test_dataset, verbose=0)
```

Predictions are collected:

```python
probabilities = model.predict(images, verbose=0).reshape(-1)
y_pred_array = (y_probability_array >= 0.5).astype(int)
```

Then the code prints:

```python
classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
```

The classification report includes:

- precision
- recall
- F1-score
- support
- accuracy

---

## 18. Confusion Matrix

The confusion matrix shows where the model is correct and wrong.

Code:

```python
matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
```

For this project:

```text
0 = no_cavity
1 = cavity
```

The matrix can be read like this:

```text
                Pred no_cavity | Pred cavity
True no_cavity       TN        |     FP
True cavity          FN        |     TP
```

Saved output:

```text
outputs/confusion_matrix.png
```

---

## 19. Prediction Visualization

The code saves a 3x3 grid of prediction examples:

```text
outputs/predictions.png
```

Each image title shows:

```text
Predicted class
True class
Confidence
```

Title color:

```text
green = correct prediction
red   = wrong prediction
```

This is useful for presentation because the TA can see real example predictions.

---

## 20. Training History Plots

The project saves:

```text
outputs/phase1_history.png
outputs/phase2_history.png
outputs/training_history.png
```

These plots show:

- training accuracy
- validation accuracy
- training loss
- validation loss

Why they matter:

- If training accuracy is high but validation accuracy is low, the model is overfitting.
- If both are low, the model is underfitting.
- If validation improves after fine-tuning, Phase 2 helped.

---

## 21. Main Function Sequence

The project starts from:

```python
if __name__ == "__main__":
    main()
```

Inside `main`, the simplified sequence is:

```python
args = parse_args()
set_reproducibility()

ensure_dataset(args.dataset)
train_dataset, test_dataset, class_names, class_weight = load_datasets(args.dataset, args.batch_size)

data_augmentation = build_augmentation()
save_augmented_preview(train_dataset, data_augmentation)
augmented_train_dataset = apply_training_augmentation(train_dataset, data_augmentation)

model, base_model = build_model(args.weights)
callbacks = make_callbacks()

history_phase1 = train_head(...)
history_phase2 = fine_tune(...)

combined_history = combine_histories(history_phase1, history_phase2)
save_combined_history_plot(combined_history, phase1_epochs)

_, test_accuracy = evaluate_model(test_dataset, class_names)
print_summary(combined_history, test_accuracy)
```

This makes the code easy to explain because each function has one clear job.

---

## 22. Command-Line Options

The code supports command-line arguments:

```python
parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
parser.add_argument("--epochs-head", type=int, default=EPOCHS_HEAD)
parser.add_argument("--epochs-finetune", type=int, default=EPOCHS_FINETUNE)
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
parser.add_argument("--skip-train", action="store_true")
```

Examples:

Run normally:

```powershell
python dental_xray_detection.py
```

Run using the existing TensorFlow venv:

```powershell
& "D:\8th term\NN\Project\venv\Scripts\python.exe" dental_xray_detection.py
```

Quick one-epoch check on the real dataset:

```powershell
& "D:\8th term\NN\Project\venv\Scripts\python.exe" dental_xray_detection.py --epochs-head 1 --epochs-finetune 0 --weights none
```

Use another dataset path:

```powershell
python dental_xray_detection.py --dataset dataset
```

Evaluate only after training:

```powershell
python dental_xray_detection.py --skip-train
```

---

## 23. Why The Notebook Exists

The notebook:

```text
dental_xray_detection.ipynb
```

does not duplicate all the implementation manually. Instead, it imports functions from:

```text
dental_xray_detection.py
```

This is cleaner because:

- the Python file contains the real implementation
- the notebook is easier to present step by step
- no copy-paste mismatch between notebook and script
- debugging is easier

Example notebook import:

```python
from dental_xray_detection import (
    build_augmentation,
    build_model,
    load_datasets,
    train_head,
    fine_tune,
    evaluate_model,
)
```

---

## 24. What We Used

### TensorFlow and Keras

Used for:

- image loading
- data pipeline
- MobileNetV2
- Sequential model
- training
- callbacks
- model saving/loading

### NumPy

Used for:

- label arrays
- class counts
- converting predictions

### Matplotlib

Used for:

- augmented sample grid
- training curves
- confusion matrix figure
- prediction examples

### scikit-learn

Used for:

- `classification_report`
- `confusion_matrix`

### csv and pathlib

Used for:

- reading bbox CSV labels
- clean filesystem paths

---

## 25. Why The Implementation Is Simple

The project avoids unnecessary complexity:

- no custom CNN from scratch
- no object detection training
- no YOLO setup
- no synthetic data
- no complex preprocessing folder generation
- no manual image copying

Instead, it directly reads the real TUFTS images and CSV labels.

This makes the project easier to discuss:

```text
CSV gives labels -> TensorFlow loads images -> MobileNetV2 learns classifier -> evaluate results
```

---

## 26. Important Limitation

This is a simple binary image classifier.

The CSV has bounding boxes, but the model does not predict bounding-box locations. It only predicts whether the image is cavity/no_cavity.

So in discussion:

```text
We used bounding boxes only to create image-level labels.
We did not train an object detection model.
```

This is a reasonable simplification for a neural networks college project because the focus is CNN classification and transfer learning.

---

## 27. Another Important Limitation

The test set has only 20 images:

```text
4 cavity
16 no_cavity
```

So test accuracy can change a lot with only a few mistakes.

Example:

```text
1 wrong image out of 20 = 5% accuracy drop
```

When discussing results, mention that a larger test set would give a more reliable evaluation.

---

## 28. What To Say To The TA

You can explain the project like this:

```text
This project detects whether a dental X-ray image has a cavity/anomaly.
The dataset is the TUFTS dental X-ray dataset. It contains radiographs and bounding-box CSV files.
Because our project is classification, not object detection, we convert the bbox CSV into binary labels:
if an image appears in the CSV, it is cavity; otherwise, it is no_cavity.

Then we load images with TensorFlow, resize them to 96x96 for CPU speed,
normalize pixels to 0..1, and use light data augmentation.

The model uses MobileNetV2 transfer learning. In Phase 1, MobileNetV2 is frozen
and we train only the dense classification head. In Phase 2, we unfreeze only
the last 10 layers and fine-tune with a low learning rate.

Finally, we evaluate using accuracy, classification report, confusion matrix,
training curves, and prediction examples.
```

---

## 29. Questions The TA May Ask

### Why did you use MobileNetV2?

Because it is lightweight, CPU-friendly, and pretrained. It can extract useful image features without training a large CNN from scratch.

### Why not train from scratch?

The dataset is relatively small. Training from scratch needs more data and compute. Transfer learning is better for small datasets.

### Why use binary crossentropy?

Because the task has two classes and the model output is one sigmoid neuron.

### Why use sigmoid instead of softmax?

For binary classification, one sigmoid output is enough. It outputs the probability of the positive class, which is `cavity`.

### Why resize to 96x96?

To make training faster on CPU while keeping enough visual information for a project demonstration.

### Why use class weights?

Because the dataset has more no-cavity images than cavity images. Class weights reduce majority-class bias.

### Why use augmentation?

To reduce overfitting and make the model more robust to small image variations.

### Why is the learning rate lower in fine-tuning?

Because pretrained weights should be updated carefully. A high learning rate can destroy useful pretrained features.

### Is this object detection?

No. It is image classification. Bounding boxes are used only to label images as cavity/no_cavity.

---

## 30. Final Summary

The final project is a clean CPU-friendly dental X-ray classification pipeline:

```text
Real TUFTS dataset
-> bbox CSV converted to binary labels
-> TensorFlow image pipeline
-> MobileNetV2 transfer learning
-> head training
-> last-layer fine-tuning
-> evaluation and visualizations
```

The main strengths are:

- simple implementation
- real dataset
- clear lab connection
- transfer learning
- CPU optimization
- readable workflow
- saved plots and metrics for discussion

The main limitation is:

- it performs binary classification only, not bounding-box detection
- the test set is small

For a college neural networks project, this is a strong and explainable implementation.
