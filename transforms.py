import torchvision
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandAugment(magnitude=5),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

val_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)
