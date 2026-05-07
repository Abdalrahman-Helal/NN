"""
Dental X-ray Cavity Detection - CPU-friendly transfer learning.

Workflow:
1. Load TUFTS dental X-ray dataset (images + bbox CSV -> binary labels)
2. Apply light augmentation to training data only
3. Train a frozen MobileNetV2 classification head (Phase 1)
4. Fine-tune the last 10 MobileNetV2 layers (Phase 2)
5. Evaluate with plots and a classification report

Run (normal):
    python dental_xray_detection.py

Run (custom dataset path):
    python dental_xray_detection.py --dataset dataset

Quick smoke test:
    python dental_xray_detection.py --epochs-head 1 --epochs-finetune 0 --weights none

Evaluate only (after training):
    python dental_xray_detection.py --skip-train
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


# ── Constants ─────────────────────────────────────────────────────────────────

IMG_SIZE       = (96, 96)
BATCH_SIZE     = 16
EPOCHS_HEAD    = 5
EPOCHS_FINETUNE = 3
SEED           = 42

DATASET_PATH    = Path("dataset")          # relative path; works on any machine
MODEL_DIR       = Path("models")
OUTPUT_DIR      = Path("outputs")
BEST_MODEL_PATH = MODEL_DIR / "best_model.keras"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dental X-ray cavity detection.")
    parser.add_argument("--dataset",         type=Path, default=DATASET_PATH)
    parser.add_argument("--epochs-head",     type=int,  default=EPOCHS_HEAD)
    parser.add_argument("--epochs-finetune", type=int,  default=EPOCHS_FINETUNE)
    parser.add_argument("--batch-size",      type=int,  default=BATCH_SIZE)
    parser.add_argument("--weights",         choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--skip-train",      action="store_true")
    return parser.parse_args()


def set_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ── Dataset ────────────────────────────────────────────────────────────────────

def find_tufts_root(dataset_dir: Path) -> Path | None:
    """Return the TUFTS root if the expected folder/CSV layout is found."""
    for candidate in (dataset_dir, dataset_dir / "TUFTS"):
        if (
            (candidate / "Radiographs" / "training_images").exists()
            and (candidate / "Radiographs" / "testing_images").exists()
            and (candidate / "bboxes" / "trainBoundryBoxes.csv").exists()
            and (candidate / "bboxes" / "testBoundryBoxes.csv").exists()
        ):
            return candidate
    return None


def ensure_dataset(dataset_dir: Path) -> None:
    if find_tufts_root(dataset_dir) is not None:
        return
    if (dataset_dir / "train").exists() and (dataset_dir / "test").exists():
        return
    raise FileNotFoundError(
        f"Dataset not found under: {dataset_dir}\n"
        "Expected TUFTS layout (Radiographs/ + bboxes/) or train/test class folders."
    )


def read_positive_image_ids(csv_path: Path) -> set[str]:
    """Return the set of image filenames that appear in the bbox CSV (= cavity images)."""
    ids = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            image_id = row.get("imageID", "").strip()
            if image_id:
                ids.add(image_id)
    return ids


def collect_tufts_split(tufts_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Read images and create binary labels from the bbox CSV."""
    image_dir = tufts_root / "Radiographs" / ("training_images" if split == "train" else "testing_images")
    csv_path  = tufts_root / "bboxes"       / ("trainBoundryBoxes.csv" if split == "train" else "testBoundryBoxes.csv")

    positive_ids = read_positive_image_ids(csv_path)
    image_paths  = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    labels = np.array([1 if p.name in positive_ids else 0 for p in image_paths], dtype=np.int32)
    paths  = np.array([str(p) for p in image_paths])

    print(f"{split.title()}: {len(labels)} images | {labels.sum()} cavity | {(labels == 0).sum()} no_cavity")
    return paths, labels


def load_image_from_path(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.reshape(tf.cast(label, tf.float32), (1,))
    return image, label


def make_tf_dataset(paths: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    autotune = tf.data.AUTOTUNE
    return ds.map(load_image_from_path, num_parallel_calls=autotune).batch(batch_size).cache().prefetch(autotune)


def balanced_class_weights(labels: np.ndarray) -> dict[int, float] | None:
    counts = np.bincount(labels.astype(int), minlength=2)
    if np.any(counts == 0):
        return None
    total = float(len(labels))
    return {0: total / (2.0 * counts[0]), 1: total / (2.0 * counts[1])}


def load_datasets(dataset_dir: Path, batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], dict | None]:
    tufts_root = find_tufts_root(dataset_dir)
    if tufts_root is None:
        raise RuntimeError("Only TUFTS dataset layout is supported in this version.")

    print(f"TUFTS dataset: {tufts_root.resolve()}")
    train_paths, train_labels = collect_tufts_split(tufts_root, "train")
    test_paths,  test_labels  = collect_tufts_split(tufts_root, "test")

    train_ds     = make_tf_dataset(train_paths, train_labels, batch_size, shuffle=True)
    test_ds      = make_tf_dataset(test_paths,  test_labels,  batch_size, shuffle=False)
    class_names  = ["no_cavity", "cavity"]
    class_weight = balanced_class_weights(train_labels)

    print(f"Class weights: {class_weight}")
    return train_ds, test_ds, class_names, class_weight


# ── Augmentation ───────────────────────────────────────────────────────────────

def build_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="light_augmentation",
    )


def apply_training_augmentation(train_ds: tf.data.Dataset, augmentation: tf.keras.Sequential) -> tf.data.Dataset:
    """Apply augmentation only to training batches — never to validation or test data."""
    def augment(images, labels):
        return augmentation(images, training=True), labels
    return train_ds.map(augment).prefetch(tf.data.AUTOTUNE)


def save_augmented_preview(train_ds: tf.data.Dataset, augmentation: tf.keras.Sequential) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images, _ = next(iter(train_ds.take(1)))
    image = images[0]

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for ax in axes.flat:
        aug = augmentation(tf.expand_dims(image, 0), training=True)[0]
        ax.imshow(aug)
        ax.axis("off")
    fig.suptitle("Augmented Training Samples")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "augmented_samples.png", dpi=150)
    plt.close(fig)


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(weights_choice: str) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    Lab 08 approach: pretrained MobileNetV2 backbone, frozen first, then fine-tuned.
    Augmentation lives in the dataset pipeline (not inside the model) so
    evaluate() and predict() never see random transformations.
    """
    weights = None if weights_choice == "none" else "imagenet"
    try:
        base_model = tf.keras.applications.MobileNetV2(include_top=False, weights=weights, input_shape=(*IMG_SIZE, 3))
    except Exception as e:
        print(f"Could not load ImageNet weights ({e}). Falling back to random weights.")
        base_model = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=(*IMG_SIZE, 3))

    base_model.trainable = False   # Phase 1: freeze entire backbone

    model = tf.keras.Sequential(
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    trainable = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"Total parameters:     {model.count_params():,}")
    print(f"Trainable (Phase 1):  {trainable:,}")
    return model, base_model


# ── Callbacks ──────────────────────────────────────────────────────────────────

def make_callbacks() -> list[tf.keras.callbacks.Callback]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2,
            restore_best_weights=True, verbose=1,
        ),
    ]


# ── Training ───────────────────────────────────────────────────────────────────

def train_head(model, train_ds, test_ds, callbacks, epochs, class_weight) -> tf.keras.callbacks.History | None:
    if epochs <= 0:
        return None
    print("\nPhase 1: training classification head (backbone frozen).")
    history = model.fit(train_ds, validation_data=test_ds, epochs=epochs,
                        callbacks=callbacks, class_weight=class_weight, verbose=2)
    _print_accuracy("Phase 1", history)
    _plot_history(history, "Phase 1: Head Only", OUTPUT_DIR / "phase1_history.png")
    return history


def fine_tune(model, base_model, train_ds, test_ds, callbacks, epochs, class_weight) -> tf.keras.callbacks.History | None:
    if epochs <= 0:
        return None

    print(f"\nPhase 2: fine-tuning last 10 of {len(base_model.layers)} MobileNetV2 layers.")
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    print(f"Trainable backbone layers: {sum(l.trainable for l in base_model.layers)}")

    # Low learning rate to avoid damaging pretrained weights (Lab 08 fine-tuning rule)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(train_ds, validation_data=test_ds, epochs=epochs,
                        callbacks=callbacks, class_weight=class_weight, verbose=2)
    _print_accuracy("Phase 2", history)
    _plot_history(history, "Phase 2: Fine-Tuning", OUTPUT_DIR / "phase2_history.png")
    return history


def _print_accuracy(label: str, history: tf.keras.callbacks.History) -> None:
    print(f"{label} — train: {history.history['accuracy'][-1]*100:.2f}%  "
          f"val: {history.history['val_accuracy'][-1]*100:.2f}%")


def _plot_history(history: tf.keras.callbacks.History, title: str, path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, label in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(history.history[metric],     label=f"Train {label}")
        ax.plot(history.history[f"val_{metric}"], label=f"Val {label}")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Combined history plot ──────────────────────────────────────────────────────

def combine_histories(h1, h2) -> dict[str, list[float]]:
    combined = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    for h in (h1, h2):
        if h is None:
            continue
        for key in combined:
            combined[key].extend(h.history[key])
    if not combined["accuracy"]:
        print("Warning: no training history — combined plot will be skipped.")
    return combined


def save_combined_plot(history: dict, phase1_epochs: int) -> None:
    if not history["accuracy"]:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    show_boundary = phase1_epochs > 0 and len(history["accuracy"]) > phase1_epochs

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, label in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(history[metric],          label=f"Train {label}")
        ax.plot(history[f"val_{metric}"], label=f"Val {label}")
        if show_boundary:
            ax.axvline(phase1_epochs - 0.5, color="red", linestyle="--", label="Fine-tuning starts")
        ax.set_title(f"Model {label}")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Dental X-ray Cavity Detection — Full Training History")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_history.png", dpi=150)
    plt.close(fig)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(test_ds: tf.data.Dataset, class_names: list[str]) -> tuple[float, float]:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"No saved model at: {BEST_MODEL_PATH}")

    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%  |  Test Loss: {loss:.4f}")

    # Collect all predictions
    y_true, y_prob = [], []
    for images, labels in test_ds:
        y_prob.extend(model.predict(images, verbose=0).reshape(-1))
        y_true.extend(labels.numpy().reshape(-1).astype(int))
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    _save_confusion_matrix(y_true, y_pred, class_names)
    _save_prediction_grid(model, test_ds, class_names)
    return float(loss), float(accuracy)


def _save_confusion_matrix(y_true, y_pred, class_names) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=[f"Pred {n}" for n in class_names])
    ax.set_yticks([0, 1], labels=[f"True {n}" for n in class_names])
    ax.set_title("Confusion Matrix")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(matrix[r, c]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _save_prediction_grid(model, test_ds, class_names, n: int = 9) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images, labels = next(iter(test_ds.take(1)))
    images = images[:n]
    labels = labels[:n].numpy().reshape(-1).astype(int)
    probs  = model.predict(images, verbose=0).reshape(-1)
    preds  = (probs >= 0.5).astype(int)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, true, pred, prob in zip(axes.flat, images, labels, preds, probs):
        conf  = prob if pred == 1 else 1.0 - prob
        color = "green" if pred == true else "red"
        ax.imshow(img)
        ax.set_title(f"Pred: {class_names[pred]}\nTrue: {class_names[true]}\nConf: {conf*100:.1f}%", color=color)
        ax.axis("off")
    fig.suptitle("Model Predictions on Test X-rays")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "predictions.png", dpi=150)
    plt.close(fig)


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(history: dict, test_accuracy: float) -> None:
    best_val    = max(history["val_accuracy"]) if history["val_accuracy"] else 0.0
    total_epochs = len(history["accuracy"])
    print("=" * 52)
    print("  DENTAL X-RAY DETECTION — RESULTS SUMMARY")
    print("=" * 52)
    print("  Model:         MobileNetV2 (Transfer Learning)")
    print("  Image Size:    96×96 px  (CPU optimized)")
    print(f"  Total Epochs:  {total_epochs}")
    print(f"  Best Val Acc:  {best_val*100:.2f}%")
    print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
    print("=" * 52)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    args = parse_args()
    set_reproducibility()

    ensure_dataset(args.dataset)
    train_ds, test_ds, class_names, class_weight = load_datasets(args.dataset, args.batch_size)

    if args.skip_train:
        _, test_accuracy = evaluate_model(test_ds, class_names)
        print_summary({"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}, test_accuracy)
        return

    augmentation      = build_augmentation()
    save_augmented_preview(train_ds, augmentation)
    aug_train_ds      = apply_training_augmentation(train_ds, augmentation)

    model, base_model = build_model(args.weights)
    callbacks         = make_callbacks()

    h1 = train_head(model, aug_train_ds, test_ds, callbacks, args.epochs_head,    class_weight)
    h2 = fine_tune (model, base_model,   aug_train_ds, test_ds, callbacks, args.epochs_finetune, class_weight)

    history        = combine_histories(h1, h2)
    phase1_epochs  = len(h1.history["accuracy"]) if h1 else 0
    save_combined_plot(history, phase1_epochs)

    _, test_accuracy = evaluate_model(test_ds, class_names)
    print_summary(history, test_accuracy)
    print(f"\nModel saved : {BEST_MODEL_PATH.resolve()}")
    print(f"Figures saved: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
