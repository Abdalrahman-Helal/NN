# Dental X-ray Cavity Detection

Simple CPU-friendly implementation for the neural networks project.

The code follows the project plan and the lab ideas:

- Lab 06: Keras `Sequential`, `compile`, `fit`, and `evaluate`
- Lab 07: CNN image workflow, dropout, and light data augmentation
- Lab 08: transfer learning with freeze then fine-tune

## Dataset

The default dataset path is now relative so the project can move to another laptop:

```text
dataset/
```

Inside this project folder, the script supports your TUFTS dataset layout:

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

Label rule used by the code:

```text
image listed in bbox CSV = cavity
image not listed in bbox CSV = no_cavity
```

No synthetic dataset is generated.

The script also supports a normal class-folder layout if you ever preprocess the data:

```text
data/
  train/
    no_cavity/
    cavity/
  test/
    no_cavity/
    cavity/
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python dental_xray_detection.py
```

Quick CPU smoke test on the real dataset without downloading ImageNet weights:

```bash
python dental_xray_detection.py --epochs-head 1 --epochs-finetune 0 --weights none
```

Evaluate an already saved model:

```bash
python dental_xray_detection.py --skip-train
```

Use another dataset folder:

```bash
python dental_xray_detection.py --dataset "D:\8th term\NN\Project\data"
```

## Outputs

```text
models/best_model.keras
outputs/augmented_samples.png
outputs/phase1_history.png
outputs/phase2_history.png
outputs/training_history.png
outputs/confusion_matrix.png
outputs/predictions.png
```
