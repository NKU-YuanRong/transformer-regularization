import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from src.dataset import get_dataloaders
from src.model import get_model
from src.trainer import Trainer
from src.utils import set_seed

def main(args):
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Weights & Biases for professional plotting
    # wandb.init(project="fashion-vit-group", name=args.experiment_name, config=args)

    # 2. Data (Member 2's Domain)
    train_loader, test_loader = get_dataloaders(args)

    # 3. Model
    model = get_model(args).to(device)

    # 4. Optimizer & Loss (Member 1 & 4's Domain)
    # MEMBER 1 TASK: Experiment with weight_decay here
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # MEMBER 4 TASK: Change this to LabelSmoothing if flag is set
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # MEMBER 4 TASK: Initialize Scheduler here if args.use_scheduler is True
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 5. Trainer
    trainer = Trainer(model, optimizer, criterion, device, scheduler)
    
    # 6. Run Loop
    print(f"Starting experiment: {args.experiment_name}")
    # Fit returns nothing currently; we'll modify Trainer to return metrics
    history = trainer.fit(train_loader, test_loader, epochs=args.epochs, use_mixup=args.use_mixup)

    # Save artifacts if requested
    if args.save_path:
        import os, json, datetime
        os.makedirs('./model', exist_ok=True)
        os.makedirs('./performance_log', exist_ok=True)
        model_save_file = os.path.join('./model', args.save_path)
        perf_save_file = os.path.join('./performance_log', args.save_path)

        # Save model state dict
        torch.save(model.state_dict(), model_save_file)

        # Prepare performance log (JSON for structure)
        log_payload = {
            'experiment_name': args.experiment_name,
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'epochs': args.epochs,
            'train_loss': history['train_loss'],
            'test_loss': history['test_loss'],
            'test_acc': history['test_acc'],
            'final_test_accuracy': history['test_acc'][-1] if history['test_acc'] else None
        }
        with open(perf_save_file, 'w') as f:
            json.dump(log_payload, f, indent=2)
        print(f"Saved model to {model_save_file} and performance log to {perf_save_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Basic Config
    parser.add_argument('--experiment_name', type=str, default='baseline')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)

    # MEMBER 1: Regularization
    parser.add_argument('--weight_decay', type=float, default=0.0, help="L2 Weight Decay")

    # MEMBER 2: Augmentation
    parser.add_argument('--aug_type', type=str, default='basic', 
                        choices=['basic', 'autoaugment', 'randaugment'], help="Augmentation policy")

    # MEMBER 3: MixUp
    parser.add_argument('--use_mixup', action='store_true', help="Enable MixUp training")

    # MEMBER 4: Scheduler & Loss
    parser.add_argument('--use_scheduler', action='store_true', help="Use Learning Rate Scheduler")
    parser.add_argument('--label_smoothing', type=float, default=0.0, help="Label smoothing value (e.g., 0.1)")
    parser.add_argument('--save_path', type=str, default='', help="Filename to save model parameters and performance log (stored under ./model and ./performance_log)")

    args = parser.parse_args()
    main(args)