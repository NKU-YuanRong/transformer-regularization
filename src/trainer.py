import torch
import wandb
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

    def fit(self, train_loader, test_loader, epochs, use_mixup=False):
        history = { 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [] }
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader, use_mixup)
            test_loss, test_acc = self.evaluate(test_loader)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            # MEMBER 4 TASK: Step the scheduler
            if self.scheduler:
                self.scheduler.step()

            # Log to WandB and Console
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")
            # wandb.log({"train_loss": train_loss, "test_acc": test_acc, "epoch": epoch})

        return history

    def train_one_epoch(self, loader, use_mixup):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)

            # MEMBER 3 TASK: Implement MixUp Logic here
            if use_mixup:
                # 1. Generate lambda (mixing ratio) from beta dist
                # 2. Mix images: mixed_images = lam * images + (1 - lam) * images_flipped
                # 3. Mix labels (handled in loss calculation usually)
                # For now, we pass, Member 3 needs to write this!
                pass 

            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate Loss
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
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