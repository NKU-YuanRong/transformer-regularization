# FashionMNIST Vision Transformer Training

## Overview
This project trains a Vision Transformer (ViT) on the FashionMNIST dataset. It provides a modular structure for:
- Data loading with optional augmentation policies
- A configurable ViT model (via `vit-pytorch`)
- Training loop with optional MixUp, label smoothing, and cosine LR scheduling
- Metric tracking and optional artifact saving

Directory layout:
```
main.py                # Entry point with argparse and experiment setup
requirements.txt       # Python dependencies
src/dataset.py         # DataLoader + augmentation logic
src/model.py           # Model factory returning a ViT instance
src/trainer.py         # Training & evaluation routines
src/utils.py           # Utility helpers (seed setting)
model/                 # Saved model weights (when --save_path used)
performance_log/       # Saved JSON training history (when --save_path used)
```

## Installation
Create a Python environment (recommended Python 3.11+) and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Quick Start
Run a baseline experiment for 5 epochs:
```bash
python main.py --epochs 5 --experiment_name baseline
```

Enable augmentation and scheduler:
```bash
python main.py --epochs 10 --aug_type autoaugment --use_scheduler --experiment_name aug_sched
```

Save model and performance metrics:
```bash
python main.py --epochs 5 --save_path run1.pt
```
This will create:
- `model/run1.pt` (model weights via `state_dict`)
- `performance_log/run1.pt` (JSON with train/test losses and test accuracies per epoch)

## Command Line Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--experiment_name` | str | `baseline` | Name shown in logs/W&B |
| `--epochs` | int | `30` | Number of training epochs |
| `--batch_size` | int | `64` | Batch size for train/test loaders |
| `--lr` | float | `0.001` | Learning rate for AdamW optimizer |
| `--weight_decay` | float | `0.0` | L2 regularization strength |
| `--aug_type` | str | `basic` | Data augmentation policy: `basic`, `autoaugment`, `randaugment` (placeholder) |
| `--use_mixup` | flag | False | Enables MixUp (to be implemented) |
| `--use_scheduler` | flag | False | Enables cosine annealing scheduler |
| `--label_smoothing` | float | `0.0` | Label smoothing value (e.g. `0.1`) |
| `--save_path` | str | `` | If set, saves model & metrics under `./model/<name>` and `./performance_log/<name>` |

## Saved Performance Log Format
Example JSON stored in `performance_log/run1.pt`:
```json
{
  "experiment_name": "baseline",
  "timestamp": "2025-11-22T12:00:00Z",
  "epochs": 5,
  "train_loss": [1.45, 0.92, 0.73, 0.61, 0.55],
  "test_loss": [1.20, 0.95, 0.80, 0.75, 0.70],
  "test_acc": [64.3, 70.1, 73.5, 74.2, 75.0],
  "final_test_accuracy": 75.0
}
```
(Values above are illustrative only.)

## Weights & Biases (Optional)
W&B integration is scaffolded but currently commented out in `main.py`. To enable:
1. `wandb login`
2. Uncomment `wandb.init(...)` and relevant `wandb.log(...)` lines.

## Future Work / TODOs
- Implement MixUp logic in `Trainer.train_one_epoch` when `--use_mixup` is set.
- Add RandAugment support for `--aug_type randaugment`.
- Make ViT hyperparameters configurable via argparse.
- Early stopping and checkpointing best validation accuracy.
- Add unit tests for data pipeline and training loop.
- Enhance plotting script for smoothing and interactive backend.

## Plotting Curves
You can visualize loss and accuracy across experiments using the helper script:
```bash
python scripts/draw_curve.py run1.pt run2.pt
```
Arguments can omit the `.pt` extension:
```bash
python scripts/draw_curve.py run1 run2
```
The script reads each JSON log from `performance_log/` and produces a figure with:
- Train/Test Loss (left)
- Train/Test Accuracy (right)

It saves the figure as `performance_log/curves_<timestamp>.png` and also displays it.

## Troubleshooting
| Issue | Possible Fix |
|-------|--------------|
| CUDA not used | Ensure PyTorch installed with GPU support; check `torch.cuda.is_available()` |
| Slow data loading | Reduce `num_workers` or batch size in low-resource environments |
| File save permission errors | Ensure you have write access to project directory |

## Citation
If you use `vit-pytorch`, consider citing the original ViT paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".

## License
This project is for educational use within the group. Add a license if distributing publicly.
