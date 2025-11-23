import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(args):
    """
    Creates dataloaders with the specific augmentation strategy chosen by the user.
    """

    # Base transforms (Normalization is always required)
    base_transform_list = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]

    # Augment: Start with basic Geometric transforms (PIL Input -> PIL Output)
    train_transform_list = [
        transforms.Resize((32, 32))
    ]

    # Logic: These must happen AFTER Resize but BEFORE ToTensor
    if args.aug_type == 'autoaugment':
        # AutoAugment: Learns the best augmentation policy (using CIFAR10 policy for small images)
        train_transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))

    elif args.aug_type == 'randaugment':
        # RandAugment: Randomly applies 'num_ops' operations with magnitude 'magnitude'
        # This is often more effective/robust than AutoAugment
        train_transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))

    # Augment: Add the necessary Conversion transforms (PIL Output -> Tensor)
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if args.aug_type == 'autoaugment' or args.aug_type == 'randaugment':
        # Compose the final training transform, using augmentation
        train_transform = transforms.Compose(train_transform_list)
    else:
        # baseline transform
        train_transform = transforms.Compose(base_transform_list)

    # Test transform should never have augmentation (except resize/norm)
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader