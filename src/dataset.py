
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_dataloaders(args):
    """
    Creates dataloaders with a Train/Validation/Test split.

    CRITICAL:
    - If no aug-type argument, use baseline dataset without data augmentation
    - Training set applies data augmentation (if selected).
    - Validation set does NOT apply augmentation (uses clean images).
    - Test set is reserved for final evaluation only.
    """

    # -----------------------------------------------------------
    # 1. Define Transforms
    # -----------------------------------------------------------

    # A. Build the Training Transform (Potential Augmentation)
    # Start with PIL-level transforms
    train_transform_list = [
        transforms.Resize((32, 32))
    ]

    # Logic: Augmentations must happen AFTER Resize but BEFORE ToTensor
    if args.aug_type == 'autoaugment':
        print(f"[Data] Applying AutoAugment (CIFAR10 Policy)")
        train_transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    elif args.aug_type == 'randaugment':
        print(f"[Data] Applying RandAugment")
        train_transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))

    # Add final conversion steps
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_transform = transforms.Compose(train_transform_list)

    # B. Build the Evaluation Transform (Validation/Test)
    # Pure clean data: Resize -> Tensor -> Normalize
    eval_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # -----------------------------------------------------------
    # 2. Load Datasets
    # -----------------------------------------------------------

    # We load the training source data TWICE to ensure data hygiene:
    # 1. Source with Augmentation (for the final Training subset)
    train_source_aug = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)

    # 2. Source without Augmentation (for the Validation subset)
    train_source_clean = datasets.FashionMNIST(root='./data', train=True, download=True, transform=eval_transform)

    # Load separate Test set
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=eval_transform)

    # -----------------------------------------------------------
    # 3. Create Train/Val Split (1:9 Ratio)
    # -----------------------------------------------------------
    num_total_train = len(train_source_aug)  # 60,000
    indices = list(range(num_total_train))

    # Split ratio: 10% for Validation, 90% for Training
    val_split = int(np.floor(0.1 * num_total_train))  # 6,000 images

    # Shuffle indices to ensure random split
    # Note: Assuming set_seed() was called in main.py for reproducibility
    np.random.shuffle(indices)

    # Slice the indices
    train_idx, val_idx = indices[val_split:], indices[:val_split]

    # Create Subsets
    # Train subset uses the AUGMENTED source
    train_subset = Subset(train_source_aug, train_idx)
    # Validation subset uses the CLEAN source
    val_subset = Subset(train_source_clean, val_idx)

    print(f"[Data] Split Summary: Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_dataset)}")

    # -----------------------------------------------------------
    # 4. Create DataLoaders
    # -----------------------------------------------------------
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
