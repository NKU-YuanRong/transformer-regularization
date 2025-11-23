import torch
import wandb
from tqdm import tqdm
from src.utils import mixup_data, mixup_criterion

class Trainer:
    def __init__(self, model, optimizer, criterion, device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

    def fit(self, train_loader, test_loader, epochs, use_mixup=False, mixup_alpha=0.4):
        history = { 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [] }
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader, use_mixup, mixup_alpha)
            val_loss, val_acc = self.evaluate(test_loader)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # MEMBER 4 TASK: Step the scheduler
            if self.scheduler:
                self.scheduler.step()

            # Log to WandB and Console
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Test Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
            # wandb.log({"train_loss": train_loss, "val_acc": val_acc, "epoch": epoch})

        return history

    def train_one_epoch(self, loader, use_mixup, mixup_alpha=0.4):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)

            # MEMBER 3 TASK: Implement MixUp Logic here
            if use_mixup:
                images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=mixup_alpha, device=self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate Loss (support MixUp loss when enabled)
            if use_mixup:
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # When using mixup, `labels` may not correspond to the mixed targets;
            # we approximate accuracy using targets_a which is one of the paired targets.
            if use_mixup:
                correct += (predicted == targets_a).sum().item()
            else:
                correct += (predicted == labels).sum().item()

        return running_loss / len(loader), 100 * correct / total

    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return running_loss / len(loader), 100 * correct / total
