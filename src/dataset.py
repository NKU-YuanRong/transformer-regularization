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

    # MEMBER 2 TASK: Insert Augmentations here based on args.aug_type
    # The logic is: Raw Image -> Augmentation -> Tensor -> Normalize
    if args.aug_type == 'autoaugment':
        # We insert AutoAugment at the beginning
        base_transform_list.insert(0, transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    elif args.aug_type == 'randaugment':
        # TODO: Member 2 implement RandAugment here
        pass

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