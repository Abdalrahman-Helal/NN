"""
Dental X-ray Cavity Detection - simple CPU-friendly implementation.

This follows the project plan:
1. load dental X-ray folders,
2. apply light augmentation,
3. train a frozen MobileNetV2 head,
4. fine-tune the last 10 backbone layers,
5. evaluate with plots and a classification report.

Run:
    python dental_xray_detection.py

Use a real dataset:
    python dental_xray_detection.py --dataset dataset

Quick smoke test:
    python dental_xray_detection.py --epochs-head 1 --epochs-finetune 0 --weights none
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


IMG_SIZE = (96, 96)
BATCH_SIZE = 16
EPOCHS_HEAD = 5
EPOCHS_FINETUNE = 3
SEED = 42

# Relative path so the script runs after submission on any machine; override with --dataset.
DATASET_PATH = Path("dataset")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
BEST_MODEL_PATH = MODEL_DIR / "best_model.keras"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
NORMAL_ALIASES = {"normal", "healthy", "no_cavity", "no-cavity", "no cavity", "without_cavity"}
CAVITY_ALIASES = {"cavity", "caries", "decay", "with_cavity", "with-cavity"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple dental X-ray cavity detection project.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Dataset folder. Default: ./dataset")
    parser.add_argument("--epochs-head", type=int, default=EPOCHS_HEAD, help="Epochs for frozen-base training.")
    parser.add_argument("--epochs-finetune", type=int, default=EPOCHS_FINETUNE, help="Epochs for fine-tuning.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet", help="MobileNetV2 weights.")
    parser.add_argument("--skip-train", action="store_true", help="Load models/best_model.keras and evaluate only.")
    return parser.parse_args()


def set_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def is_normal_class(name: str) -> bool:
    value = normalize_name(name)
    return value in NORMAL_ALIASES or "normal" in value or "healthy" in value or value.startswith("no_")


def is_cavity_class(name: str) -> bool:
    value = normalize_name(name)
    if value in CAVITY_ALIASES:
        return True
    blocked = value.startswith("no_") or value.startswith("without")
    return "cavity" in value and not blocked


def detect_class_names(split_dir: Path) -> list[str]:
    """Return class order as [normal_class, cavity_class] so sigmoid 1 means cavity."""
    class_dirs = [path for path in split_dir.iterdir() if path.is_dir()]
    normal = [path.name for path in class_dirs if is_normal_class(path.name)]
    cavity = [path.name for path in class_dirs if is_cavity_class(path.name)]

    if len(normal) == 1 and len(cavity) == 1:
        return [normal[0], cavity[0]]

    names = sorted(path.name for path in class_dirs)
    if len(names) == 2:
        print("Could not confidently detect class names, using sorted order:", names)
        return names

    raise ValueError(
        "Expected exactly two class folders in "
        f"{split_dir}. Use names like normal/ and cavity/."
    )


def find_tufts_root(dataset_dir: Path) -> Path | None:
    for candidate in (dataset_dir, dataset_dir / "TUFTS"):
        has_images = (candidate / "Radiographs" / "training_images").exists()
        has_test_images = (candidate / "Radiographs" / "testing_images").exists()
        has_train_csv = (candidate / "bboxes" / "trainBoundryBoxes.csv").exists()
        has_test_csv = (candidate / "bboxes" / "testBoundryBoxes.csv").exists()
        if has_images and has_test_images and has_train_csv and has_test_csv:
            return candidate
    return None


def ensure_dataset(dataset_dir: Path) -> None:
    if find_tufts_root(dataset_dir) is not None:
        return

    if (dataset_dir / "train").exists() and (dataset_dir / "test").exists():
        return

    raise FileNotFoundError(
        "Missing real dataset. Expected either TUFTS/Radiographs with bbox CSVs "
        f"or train/test class folders under: {dataset_dir}"
    )


def load_datasets(
    dataset_dir: Path,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], dict[int, float] | None]:
    tufts_root = find_tufts_root(dataset_dir)
    if tufts_root is not None:
        return load_tufts_datasets(tufts_root, batch_size)
    return load_folder_datasets(dataset_dir, batch_size)


def load_folder_datasets(
    dataset_dir: Path,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], dict[int, float] | None]:
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    class_names = detect_class_names(train_dir)
    class_weight = class_weights_from_folders(train_dir, class_names)

    print(f"Class order: 0 = {class_names[0]}, 1 = {class_names[1]}")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=class_names,
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True,
        seed=SEED,
    )
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        class_names=class_names,
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    def scale_pixels(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return tf.cast(images, tf.float32) / 255.0, labels

    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.map(scale_pixels).cache().prefetch(autotune)
    test_dataset = test_dataset.map(scale_pixels).cache().prefetch(autotune)

    print(f"Training batches: {tf.data.experimental.cardinality(train_dataset).numpy()}")
    print(f"Test batches:     {tf.data.experimental.cardinality(test_dataset).numpy()}")
    print(f"Class weights:    {class_weight}")
    return train_dataset, test_dataset, class_names, class_weight


def read_positive_image_ids(csv_path: Path) -> set[str]:
    positive_ids = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            image_id = row.get("imageID", "").strip()
            if image_id:
                positive_ids.add(image_id)
    return positive_ids


def collect_tufts_split(tufts_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    image_folder_name = "training_images" if split == "train" else "testing_images"
    csv_name = "trainBoundryBoxes.csv" if split == "train" else "testBoundryBoxes.csv"
    image_dir = tufts_root / "Radiographs" / image_folder_name
    csv_path = tufts_root / "bboxes" / csv_name

    positive_ids = read_positive_image_ids(csv_path)
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    labels = np.array([1 if path.name in positive_ids else 0 for path in image_paths], dtype=np.int32)
    paths = np.array([str(path) for path in image_paths])

    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    print(f"{split.title()} images: {len(labels)} total | {positives} cavity | {negatives} no_cavity")
    return paths, labels


def load_image_from_path(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.reshape(tf.cast(label, tf.float32), (1,))
    return image, label


def make_path_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    autotune = tf.data.AUTOTUNE
    return dataset.map(load_image_from_path, num_parallel_calls=autotune).batch(batch_size).cache().prefetch(autotune)


def load_tufts_datasets(
    tufts_root: Path,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], dict[int, float] | None]:
    print(f"Using TUFTS dataset: {tufts_root.resolve()}")
    print("Label rule: image has any bbox row = cavity, otherwise no_cavity")

    train_paths, train_labels = collect_tufts_split(tufts_root, "train")
    test_paths, test_labels = collect_tufts_split(tufts_root, "test")

    train_dataset = make_path_dataset(train_paths, train_labels, batch_size=batch_size, shuffle=True)
    test_dataset = make_path_dataset(test_paths, test_labels, batch_size=batch_size, shuffle=False)
    class_names = ["no_cavity", "cavity"]
    class_weight = balanced_class_weights(train_labels)

    print(f"Training batches: {tf.data.experimental.cardinality(train_dataset).numpy()}")
    print(f"Test batches:     {tf.data.experimental.cardinality(test_dataset).numpy()}")
    print(f"Class weights:    {class_weight}")
    return train_dataset, test_dataset, class_names, class_weight


def class_weights_from_folders(train_dir: Path, class_names: list[str]) -> dict[int, float] | None:
    labels = []
    for class_index, class_name in enumerate(class_names):
        class_dir = train_dir / class_name
        count = sum(
            1 for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels.extend([class_index] * count)

    if not labels:
        return None
    return balanced_class_weights(np.array(labels, dtype=np.int32))


def balanced_class_weights(labels: np.ndarray) -> dict[int, float] | None:
    counts = np.bincount(labels.astype(int), minlength=2)
    if np.any(counts == 0):
        return None

    total = float(len(labels))
    return {
        0: total / (2.0 * float(counts[0])),
        1: total / (2.0 * float(counts[1])),
    }


def build_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="light_augmentation",
    )


def apply_training_augmentation(
    train_dataset: tf.data.Dataset,
    data_augmentation: tf.keras.Sequential,
) -> tf.data.Dataset:
    """Apply augmentation only to training batches, never validation/test batches."""
    autotune = tf.data.AUTOTUNE

    def augment(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return data_augmentation(images, training=True), labels

    # No num_parallel_calls: light aug + CPU; prefetch batches matter more than threaded map overhead.
    return train_dataset.map(augment).prefetch(autotune)


def save_augmented_preview(train_dataset: tf.data.Dataset, data_augmentation: tf.keras.Sequential) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images, _ = next(iter(train_dataset.take(1)))
    image = images[0]

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for axis in axes.flat:
        augmented = data_augmentation(tf.expand_dims(image, axis=0), training=True)[0]
        axis.imshow(augmented)
        axis.axis("off")

    fig.suptitle("Augmented Training Samples")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "augmented_samples.png", dpi=150)
    plt.close(fig)


def load_mobilenet_base(weights_choice: str) -> tf.keras.Model:
    weights = None if weights_choice == "none" else "imagenet"
    try:
        return tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=weights,
            input_shape=(*IMG_SIZE, 3),
        )
    except Exception as error:
        if weights == "imagenet":
            print("Could not load ImageNet weights. Falling back to weights=None.")
            print(f"Reason: {error}")
            return tf.keras.applications.MobileNetV2(
                include_top=False,
                weights=None,
                input_shape=(*IMG_SIZE, 3),
            )
        raise


def build_model(weights_choice: str) -> tuple[tf.keras.Model, tf.keras.Model]:
    # Lab 08 idea: use a pretrained CNN base, freeze it first, then train a small new head.
    # MobileNetV2 is chosen because its depthwise convolutions are light enough for CPU.
    # Augmentation is not part of this graph; it is applied only to the train dataset via
    # apply_training_augmentation() so evaluate() / predict() never see random aug.
    base_model = load_mobilenet_base(weights_choice)
    base_model.trainable = False

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

    dental_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    dental_model.summary()
    print(f"Total parameters: {dental_model.count_params():,}")
    print(f"Trainable parameters (Phase 1): {count_trainable_parameters(dental_model):,}")
    return dental_model, base_model


def count_trainable_parameters(model: tf.keras.Model) -> int:
    return int(sum(np.prod(variable.shape) for variable in model.trainable_variables))


def make_callbacks() -> list[tf.keras.callbacks.Callback]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
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


def train_head(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    callbacks: list[tf.keras.callbacks.Callback],
    epochs: int,
    class_weight: dict[int, float] | None,
) -> tf.keras.callbacks.History | None:
    if epochs <= 0:
        return None

    print("\nPhase 1: training the small classification head only.")
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )
    print_final_accuracy("Phase 1", history)
    plot_history(history, "Phase 1 Training: Head Only", OUTPUT_DIR / "phase1_history.png")
    return history


def fine_tune(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    callbacks: list[tf.keras.callbacks.Callback],
    epochs: int,
    class_weight: dict[int, float] | None,
) -> tf.keras.callbacks.History | None:
    if epochs <= 0:
        return None

    print("\nPhase 2: fine-tuning only the last 10 MobileNetV2 layers.")
    print(f"MobileNetV2 layers: {len(base_model.layers)}")

    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    trainable_layers = sum(layer.trainable for layer in base_model.layers)
    print(f"Trainable MobileNetV2 layers: {trainable_layers}")

    # Lab 08 fine-tuning rule: use a very small learning rate so pretrained weights are not damaged.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )
    print_final_accuracy("Phase 2", history)
    plot_history(history, "Phase 2 Fine-Tuning", OUTPUT_DIR / "phase2_history.png")
    return history


def print_final_accuracy(label: str, history: tf.keras.callbacks.History) -> None:
    train_accuracy = history.history["accuracy"][-1]
    validation_accuracy = history.history["val_accuracy"][-1]
    print(f"{label} final train accuracy: {train_accuracy * 100:.2f}%")
    print(f"{label} final validation accuracy: {validation_accuracy * 100:.2f}%")


def plot_history(history: tf.keras.callbacks.History, title: str, output_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Training Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Training Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def combine_histories(
    history_phase1: tf.keras.callbacks.History | None,
    history_phase2: tf.keras.callbacks.History | None,
) -> dict[str, list[float]]:
    combined = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    for history in (history_phase1, history_phase2):
        if history is None:
            continue
        for key in combined:
            combined[key].extend(history.history[key])
    if not combined["accuracy"]:
        print(
            "Warning: No training history to combine (e.g. --epochs-head 0 and --epochs-finetune 0). "
            "Combined history plot and epoch-based summary will be empty or skipped."
        )
    return combined


def save_combined_history_plot(history: dict[str, list[float]], phase1_epochs: int) -> None:
    if not history["accuracy"]:
        print("Combined history plot skipped: no epoch metrics (both training phases may have been skipped).")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["accuracy"], label="Training Accuracy")
    axes[0].plot(history["val_accuracy"], label="Validation Accuracy")
    show_finetune_boundary = phase1_epochs > 0 and len(history["accuracy"]) > phase1_epochs
    if show_finetune_boundary:
        axes[0].axvline(phase1_epochs - 0.5, color="red", linestyle="--", label="Fine-tuning starts")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["loss"], label="Training Loss")
    axes[1].plot(history["val_loss"], label="Validation Loss")
    if show_finetune_boundary:
        axes[1].axvline(phase1_epochs - 0.5, color="red", linestyle="--", label="Fine-tuning starts")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Dental X-ray Cavity Detection - Full Training History")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_history.png", dpi=150)
    plt.close(fig)


def collect_predictions(model: tf.keras.Model, test_dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = []
    y_probability = []

    for images, labels in test_dataset:
        probabilities = model.predict(images, verbose=0).reshape(-1)
        y_probability.extend(probabilities.tolist())
        y_true.extend(labels.numpy().reshape(-1).astype(int).tolist())

    y_true_array = np.array(y_true, dtype=int)
    y_probability_array = np.array(y_probability, dtype=float)
    y_pred_array = (y_probability_array >= 0.5).astype(int)
    return y_true_array, y_pred_array, y_probability_array


def evaluate_model(test_dataset: tf.data.Dataset, class_names: list[str]) -> tuple[float, float]:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing saved model: {BEST_MODEL_PATH}")

    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    loss, accuracy = best_model.evaluate(test_dataset, verbose=0)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}% | Test Loss: {loss:.4f}")

    y_true, y_pred, _ = collect_predictions(best_model, test_dataset)
    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    save_confusion_matrix(y_true, y_pred, class_names)
    save_prediction_examples(best_model, test_dataset, class_names)
    return float(loss), float(accuracy)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=[f"Pred {name}" for name in class_names])
    axis.set_yticks([0, 1], labels=[f"True {name}" for name in class_names])
    axis.set_title("Confusion Matrix")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_prediction_examples(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    class_names: list[str],
    max_images: int = 9,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images, labels = next(iter(test_dataset.take(1)))
    images = images[:max_images]
    labels = labels[:max_images].numpy().reshape(-1).astype(int)
    probabilities = model.predict(images, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for axis in axes.flat:
        axis.axis("off")

    for axis, image, true_label, predicted_label, probability in zip(
        axes.flat,
        images,
        labels,
        predictions,
        probabilities,
    ):
        confidence = probability if predicted_label == 1 else 1.0 - probability
        title_color = "green" if predicted_label == true_label else "red"
        axis.imshow(image)
        axis.set_title(
            f"Pred: {class_names[predicted_label]}\n"
            f"True: {class_names[true_label]}\n"
            f"Conf: {confidence * 100:.1f}%",
            color=title_color,
        )
        axis.axis("off")

    fig.suptitle("Model Predictions on Test X-rays")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "predictions.png", dpi=150)
    plt.close(fig)


def print_summary(history: dict[str, list[float]], test_accuracy: float) -> None:
    best_validation = max(history["val_accuracy"]) if history["val_accuracy"] else 0.0
    total_epochs = len(history["accuracy"])

    print("=" * 52)
    print("  DENTAL X-RAY DETECTION - RESULTS SUMMARY")
    print("=" * 52)
    print("  Model:         MobileNetV2")
    print("  Image Size:    96x96 px")
    print("  Training:      CPU-friendly transfer learning")
    print(f"  Total Epochs:  {total_epochs}")
    print(f"  Best Val Acc:  {best_validation * 100:.2f}%")
    print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")
    print("=" * 52)


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    args = parse_args()
    set_reproducibility()

    ensure_dataset(args.dataset)
    train_dataset, test_dataset, class_names, class_weight = load_datasets(args.dataset, args.batch_size)

    if args.skip_train:
        _, test_accuracy = evaluate_model(test_dataset, class_names)
        print_summary({"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}, test_accuracy)
        return

    data_augmentation = build_augmentation()
    save_augmented_preview(train_dataset, data_augmentation)
    augmented_train_dataset = apply_training_augmentation(train_dataset, data_augmentation)

    model, base_model = build_model(args.weights)
    callbacks = make_callbacks()

    history_phase1 = train_head(
        model,
        augmented_train_dataset,
        test_dataset,
        callbacks,
        args.epochs_head,
        class_weight,
    )
    history_phase2 = fine_tune(
        model,
        base_model,
        augmented_train_dataset,
        test_dataset,
        callbacks,
        args.epochs_finetune,
        class_weight,
    )

    combined_history = combine_histories(history_phase1, history_phase2)
    phase1_epochs = len(history_phase1.history["accuracy"]) if history_phase1 else 0
    save_combined_history_plot(combined_history, phase1_epochs)

    _, test_accuracy = evaluate_model(test_dataset, class_names)
    print_summary(combined_history, test_accuracy)
    print(f"\nSaved model: {BEST_MODEL_PATH.resolve()}")
    print(f"Saved figures: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
