# Dental X-ray Cavity Detection — Project Explanation

This document explains the whole project in a discussion-friendly way:
what the project does, why each choice was made, how the code works,
and what to say when the TA asks questions.

Main file: `dental_xray_detection.py`

---

## 1. Project Idea

The goal is to classify dental X-ray images into two classes:

```
0 = no_cavity
1 = cavity
```

The model receives a dental X-ray image and predicts whether it contains a cavity or anomaly.

We use MobileNetV2 transfer learning — a pretrained CNN backbone with a small new classifier head trained on our dental dataset. This is the approach from **Lab 08**.

---

## 2. Why This Fits The Neural Networks Course

| Lab | Concept | Where It Appears |
|-----|---------|-----------------|
| Lab 06 | Keras Sequential, `compile`, `fit`, `evaluate` | Model build and training loop |
| Lab 07 | CNN image classification, Dropout, Data augmentation | MobileNetV2 backbone + augmentation pipeline |
| Lab 08 | Transfer learning, freeze/unfreeze | Phase 1 (frozen) and Phase 2 (fine-tune last 10 layers) |

---

## 3. Project Files

```
dental_xray_detection.py     main implementation
PROJECT_EXPLANATION.md       this file
requirements.txt             required Python packages
dataset/                     TUFTS dental X-ray dataset (copy here before running)
models/                      created after training — contains best_model.keras
outputs/                     created after training — contains all plots
```

After training the code saves:

```
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

---

## 4. Dataset Structure (TUFTS)

```
dataset/
  TUFTS/
    Radiographs/
      training_images/     980 panoramic X-ray images
      testing_images/       20 panoramic X-ray images
    bboxes/
      trainBoundryBoxes.csv
      testBoundryBoxes.csv
```

The CSV files contain bounding-box annotations, for example:

```
imageID,class,x-min,y-min,width,height
149.JPG,4,651,623,83,61
727.JPG,2,1081,592,50,59
```

---

## 5. Labeling Rule

This project is **image classification**, not object detection.
The bounding boxes are used only to create binary image labels:

```
Image appears in bbox CSV  →  label = cavity   (1)
Image not in bbox CSV      →  label = no_cavity (0)
```

Code that reads the CSV:

```python
def read_positive_image_ids(csv_path: Path) -> set[str]:
    ids = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            image_id = row.get("imageID", "").strip()
            if image_id:
                ids.add(image_id)
    return ids
```

Code that creates the labels:

```python
labels = np.array([1 if p.name in positive_ids else 0 for p in image_paths], dtype=np.int32)
```

Dataset statistics:

```
Train: 980 images | 335 cavity | 645 no_cavity
Test:   20 images |   4 cavity |  16 no_cavity
```

---

## 6. Why We Use Class Weights

The training set is imbalanced: more no-cavity images than cavity images.
Without correction, the model tends to predict the majority class.

Class weights give the minority class (cavity) more importance during training:

```python
def balanced_class_weights(labels):
    counts = np.bincount(labels.astype(int), minlength=2)
    total  = float(len(labels))
    return {0: total / (2.0 * counts[0]),
            1: total / (2.0 * counts[1])}
```

For this dataset the approximate weights are:

```
no_cavity → 0.76
cavity    → 1.46
```

---

## 7. Workflow Sequence

```
TUFTS images + bbox CSV
       ↓
Binary labels: no_cavity / cavity
       ↓
TensorFlow dataset (resize 96×96, normalize 0..1, cache, prefetch)
       ↓
Light data augmentation (training only)
       ↓
MobileNetV2 backbone (frozen in Phase 1)
       ↓
GlobalAveragePooling2D
       ↓
Dense(64, relu) + Dropout(0.3)
       ↓
Dense(1, sigmoid)
       ↓
Binary prediction: cavity or no_cavity
       ↓
Evaluate: accuracy, classification report, confusion matrix, plots
```

---

## 8. Constants

```python
IMG_SIZE        = (96, 96)    # small for CPU speed
BATCH_SIZE      = 16          # low memory usage on CPU
EPOCHS_HEAD     = 5           # Phase 1
EPOCHS_FINETUNE = 3           # Phase 2
SEED            = 42          # reproducibility
DATASET_PATH    = Path("dataset")   # relative — works on any machine
```

**Why 96×96?** Much faster on CPU than 224×224 while still giving enough visual information.

**Why batch size 16?** Keeps memory usage low and stable on CPU.

---

## 9. Image Loading Pipeline

```python
def load_image_from_path(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.reshape(tf.cast(label, tf.float32), (1,))
    return image, label
```

After loading, the dataset is batched, cached, and prefetched:

```python
ds.map(load_image_from_path, num_parallel_calls=autotune).batch(batch_size).cache().prefetch(autotune)
```

**Why cache and prefetch?** On CPU, image loading can slow training down.
`cache()` keeps processed batches in memory and `prefetch()` prepares the next batch while the model trains on the current one.

---

## 10. Data Augmentation

```python
def build_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ], name="light_augmentation")
```

We keep augmentation light because these are medical images — heavy transforms can distort diagnostically important features.

| Layer | Purpose |
|-------|---------|
| `RandomFlip("horizontal")` | Left/right variation |
| `RandomRotation(0.05)` | Small tilt only |
| `RandomZoom(0.1)` | Small zoom variation |

**Key design choice:** augmentation is applied via `.map()` on the training dataset only — it is **not** inside the model. This means `evaluate()` and `predict()` never see random transformations:

```python
def apply_training_augmentation(train_ds, augmentation):
    def augment(images, labels):
        return augmentation(images, training=True), labels
    return train_ds.map(augment).prefetch(tf.data.AUTOTUNE)
```

---

## 11. Why Transfer Learning?

Training a CNN from scratch needs a large dataset and strong hardware. The TUFTS test set has only 20 images.

Transfer learning reuses a CNN already trained on a large dataset (ImageNet). Even though ImageNet is not medical data, early CNN layers learn general features (edges, curves, textures) that are still useful for X-rays.

**Why MobileNetV2?**
- Lightweight and CPU-friendly
- Uses depthwise separable convolutions (~9× fewer operations than standard Conv2D)
- Pretrained on ImageNet, available in Keras

---

## 12. Model Architecture

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(96, 96, 3)),
    base_model,                              # MobileNetV2, frozen in Phase 1
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid"),
], name="dental_model")
```

| Layer | Role |
|-------|------|
| `Input(96, 96, 3)` | Receives RGB X-ray image |
| `MobileNetV2` | Extracts CNN features |
| `GlobalAveragePooling2D` | Converts feature maps to a compact vector |
| `Dense(64, relu)` | Learns task-specific patterns |
| `Dropout(0.3)` | Reduces overfitting |
| `Dense(1, sigmoid)` | Outputs cavity probability |

**Why sigmoid?** Binary classification — one output neuron is enough.
Output ≥ 0.5 → cavity. Output < 0.5 → no_cavity.

**Phase 1 compilation:**

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
```

---

## 13. Training Strategy

### Phase 1 — Train Head Only (backbone frozen)

```python
base_model.trainable = False
```

Only the Dense layers are updated. This is fast and safe — it cannot damage pretrained features.

### Phase 2 — Fine-Tune Last 10 Layers

```python
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False
```

Then recompile with a much smaller learning rate:

```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), ...)
```

**Why low learning rate in Phase 2?** Pretrained weights must be updated carefully. A high learning rate destroys useful features. `1e-5` is 100× smaller than Phase 1.

---

## 14. Callbacks

```python
ModelCheckpoint(monitor="val_accuracy", save_best_only=True)
EarlyStopping  (monitor="val_loss",     patience=2, restore_best_weights=True)
```

- **ModelCheckpoint** — saves `models/best_model.keras` whenever validation accuracy improves.
- **EarlyStopping (patience=2)** — stops training if validation loss does not improve for 2 epochs, saving CPU time and reducing overfitting.

---

## 15. Evaluation

After training, the best saved model is loaded and evaluated:

```python
model = tf.keras.models.load_model(BEST_MODEL_PATH)
loss, accuracy = model.evaluate(test_ds, verbose=0)
```

Predictions use a 0.5 threshold:

```python
y_pred = (y_prob >= 0.5).astype(int)
```

The classification report prints precision, recall, F1-score, and accuracy.
The confusion matrix shows true positives, false positives, false negatives, and true negatives.

---

## 16. Output Plots

| File | Contents |
|------|---------|
| `augmented_samples.png` | 3×3 grid of the same image with different augmentations |
| `phase1_history.png` | Accuracy and loss curves for Phase 1 |
| `phase2_history.png` | Accuracy and loss curves for Phase 2 |
| `training_history.png` | Combined Phase 1 + Phase 2 with a red dashed line at the fine-tuning boundary |
| `confusion_matrix.png` | 2×2 confusion matrix |
| `predictions.png` | 3×3 grid of test predictions (green = correct, red = wrong) |

---

## 17. Command-Line Options

| Argument | Default | Purpose |
|----------|---------|---------|
| `--dataset` | `dataset` | Path to dataset folder |
| `--epochs-head` | `5` | Phase 1 epochs |
| `--epochs-finetune` | `3` | Phase 2 epochs |
| `--batch-size` | `16` | Batch size |
| `--weights` | `imagenet` | `imagenet` or `none` |
| `--skip-train` | off | Load saved model and evaluate only |

Examples:

```bash
# Normal run
python dental_xray_detection.py

# Quick smoke test (1 epoch, no weights download)
python dental_xray_detection.py --epochs-head 1 --epochs-finetune 0 --weights none

# Evaluate only after training
python dental_xray_detection.py --skip-train
```

---

## 18. Known Limitations

**Binary classification only.** The CSV bounding boxes are used only for labeling. The model does not predict bounding-box locations — it is a classifier, not an object detector.

**Small test set.** The test set has only 20 images (4 cavity, 16 no_cavity). One wrong prediction = 5% accuracy drop, so test accuracy numbers can vary a lot. A larger test set would give more reliable evaluation.

---

## 19. What To Say To The TA

```
This project detects whether a dental X-ray image contains a cavity.

The dataset is TUFTS dental X-rays. It has panoramic radiograph images
and bounding-box CSV files. Because our task is classification (not object
detection), we use the CSV to create binary labels: if an image appears in
the CSV it is cavity, otherwise no_cavity.

Images are loaded with TensorFlow, resized to 96×96 for CPU speed,
and normalized to 0..1. We use light augmentation — flip, small rotation,
small zoom — on training data only.

The model is MobileNetV2 transfer learning. In Phase 1 the backbone is
frozen and we train only the Dense head. In Phase 2 we unfreeze the last
10 layers and fine-tune with a learning rate of 1e-5.

We evaluate with test accuracy, classification report, confusion matrix,
training curves, and a prediction grid.
```

---

## 20. Questions The TA May Ask

**Why MobileNetV2?**
Lightweight, CPU-friendly, pretrained. Extracts useful features without training a large CNN from scratch.

**Why not train from scratch?**
Small dataset. Transfer learning is better suited for small datasets because the backbone already knows how to extract visual features.

**Why binary crossentropy?**
Two classes, one sigmoid output neuron — binary crossentropy is the correct loss for this setup.

**Why sigmoid instead of softmax?**
For binary classification one sigmoid neuron is enough. It outputs the probability of the positive class (cavity).

**Why 96×96?**
Smaller images train much faster on CPU while retaining enough visual information for a project demonstration.

**Why class weights?**
The dataset is imbalanced — more no-cavity images. Class weights reduce majority-class bias during training.

**Why low learning rate in Phase 2?**
To update pretrained weights carefully without destroying useful features.

**Is this object detection?**
No. Bounding boxes are only used to create image-level labels. The model is a binary classifier.

**Why is the test accuracy sometimes low?**
The test set has only 20 images. One wrong prediction equals a 5% drop. A larger test set would give more stable numbers.
