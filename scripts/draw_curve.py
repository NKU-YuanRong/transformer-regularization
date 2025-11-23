#!/usr/bin/env python3
"""Plot accuracy and loss curves for one or more experiments.

Usage:
    python scripts/draw_curve.py exp1.pt exp2.pt ...
    # Or without .pt extension if your saved file is just a name
    python scripts/draw_curve.py exp1 exp2

The script looks for files under ./performance_log/<name> (exact argument as given). If the
argument does not end with '.pt', it assumes the performance log was saved with that exact name.
Files are JSON containing keys: train_loss, train_acc, test_loss, test_acc.

Outputs:
    A matplotlib figure with two subplots (Loss & Accuracy) over epochs for all experiments.
    The figure is displayed and also saved to ./performance_log/curves_<timestamp>.png.
"""
import os
import sys
import json
import datetime
import matplotlib.pyplot as plt

PERF_DIR = os.path.join(os.path.dirname(__file__), '..', 'performance_log')


def resolve_log_path(arg: str) -> str:
    # Accept both names with or without extension
    candidate = os.path.join(PERF_DIR, arg)
    if os.path.isfile(candidate):
        return candidate
    # Try adding .pt if missing
    if not arg.endswith('.pt'):
        candidate_pt = os.path.join(PERF_DIR, arg + '.pt')
        if os.path.isfile(candidate_pt):
            return candidate_pt
    raise FileNotFoundError(f"Performance log file for '{arg}' not found in {PERF_DIR}.")


def load_history(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def plot_histories(histories: dict):
    # histories: name -> history dict
    plt.style.use('seaborn-v0_8')
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    for name, h in histories.items():
        epochs = range(1, len(h.get('train_loss', [])) + 1)
        if h.get('train_loss'):
            ax_loss.plot(epochs, h['train_loss'], label=f'{name} Train Loss', linestyle='-')
        if h.get('val_loss'):
            ax_loss.plot(epochs, h['val_loss'], label=f'{name} Val Loss', linestyle='--')
        if h.get('train_acc'):
            ax_acc.plot(epochs, h['train_acc'], label=f'{name} Train Acc', linestyle='-')
        if h.get('val_acc'):
            ax_acc.plot(epochs, h['val_acc'], label=f'{name} Val Acc', linestyle='--')

    ax_loss.set_title('Loss Curves')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend(fontsize=8)

    ax_acc.set_title('Accuracy Curves (%)')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.legend(fontsize=8)

    fig.tight_layout()
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(PERF_DIR, f'curves_{timestamp}.png')
    fig.savefig(out_path, dpi=150)
    print(f'Saved plot to {out_path}')
    plt.show()


def main():
    if len(sys.argv) < 2:
        print('Provide at least one performance log name (e.g. python scripts/draw_curve.py run1.pt run2.pt)')
        sys.exit(1)

    names = sys.argv[1:]
    histories = {}
    for name in names:
        path = resolve_log_path(name)
        data = load_history(path)
        # Derive display name from argument (strip extension)
        display_name = os.path.splitext(os.path.basename(name))[0]
        histories[display_name] = data
        print(f"Loaded {path}")

    plot_histories(histories)


if __name__ == '__main__':
    main()
