# Dental X-ray Cavity Detection

A neural network college project that detects cavities in dental X-rays using transfer learning on the TUFTS dataset.

## What I built

An image classification pipeline that takes dental X-ray images and predicts whether they contain a cavity or not:

- **Dataset** — TUFTS dental X-ray dataset; binary labels derived from bounding box CSVs (cavity vs. no cavity), with balanced class weights to handle imbalance
- **Augmentation** — light training-only augmentation (horizontal flip, small rotation, zoom) applied in the data pipeline so it never affects validation or test data
- **Model** — MobileNetV2 pretrained on ImageNet as a frozen backbone, with a custom classification head (GlobalAveragePooling → Dense 64 → Dropout 0.3 → sigmoid)
- **Two-phase training:**
  - **Phase 1** — train only the classification head with the backbone fully frozen
  - **Phase 2** — fine-tune the last 10 MobileNetV2 layers at a lower learning rate (1e-5) to avoid damaging pretrained weights
- **Evaluation** — accuracy, classification report, confusion matrix, prediction grid, and training history plots saved to `outputs/`

