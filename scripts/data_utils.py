import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, Dataset
import urllib.request
import numpy as np


def load_cifar(batch_size=128):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def load_mnist(batch_size=128):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def load_fashion_mnist(batch_size=128):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def load_imagenette(batch_size=128):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # Download from the official Imagenette repository
    train_dataset = ImageFolder(root="./data/imagenette/train", transform=transform)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = ImageFolder(root="./data/imagenette/val", transform=transform)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def load_flowers102(batch_size=128):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = torchvision.datasets.Flowers102(
        root="./data", split="train", download=True, transform=transform
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = torchvision.datasets.Flowers102(
        root="./data", split="test", download=True, transform=transform
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def load_food101(batch_size=128):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = torchvision.datasets.Food101(
        root="./data", split="train", download=True, transform=transform
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = torchvision.datasets.Food101(
        root="./data", split="test", download=True, transform=transform
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


class FilteredDataset(Dataset):
    def __init__(self, dataset, classes_to_select):
        """
        dataset: A standard MNIST Dataset (train or test).
        classes_to_select: A list of digits to keep, e.g. [1, 6, 9].
        """
        self.dataset = dataset
        self.classes_to_select = classes_to_select
        
        # Build a mapping from the old label (e.g. 1, 6, or 9) -> new label (0,1,2,...)
        self.class_to_newlabel = {
            old_label: new_label
            for new_label, old_label in enumerate(self.classes_to_select)
        }
        
        # Indices of the samples that belong to the desired classes
        self.indices = [
            i for i, y in enumerate(self.dataset.targets)
            #if int(y.item()) in self.classes_to_select
            if int(y) in self.classes_to_select
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get the sample (img, label) at position idx in the filtered subset.
        """
        real_index = self.indices[idx]
        img, old_label = self.dataset[real_index]  # old_label is a tensor
        
        # Convert tensor label to integer and remap
        # old_label = int(old_label.item())  # for optical_digits dataset

        new_label = self.class_to_newlabel[old_label]  # Map to range {0,...,len(classes_to_select)-1}

        return img, new_label
    

def load_mnist_subset(batch_size=128, classes_to_select=[0,1,2,3,4,5,6,7,8,9], num_samples_per_class=1000):
    """
    Loads the MNIST dataset (train and test), but only includes
    the classes specified in classes_to_select.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    
    # Download the full MNIST datasets
    full_train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    full_test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_dataset = FilteredDataset(full_train_dataset, classes_to_select)
    test_dataset  = FilteredDataset(full_test_dataset, classes_to_select)

    # Get class-wise indices for training set (ensuring the filtered dataset is indexed correctly)
    class_indices = {new_label: [] for new_label in range(len(classes_to_select))}
    
    for i, (_, label) in enumerate(train_dataset):
        class_indices[label].append(i)  # Use remapped labels from FilteredDataset

    # Select **first** `num_samples_per_class` samples from each class
    selected_indices = []
    for c in range(len(classes_to_select)):  # Work with remapped labels (0,1,...)
        selected_indices.extend(class_indices[c][:num_samples_per_class])  # Take the first `num_samples_per_class`

    # Create a fixed Subset for training
    train_subset = Subset(train_dataset, selected_indices)

    # Create DataLoaders for the subsets
    train_loader = DataLoader(
        dataset=train_subset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


class OpticalDigitsDataset(Dataset):
    def __init__(self, data_url):
        # Download the data
        data = np.loadtxt(urllib.request.urlopen(data_url), delimiter=',')
        self.X = torch.tensor(data[:, :-1], dtype=torch.float32)
        self.targets = torch.tensor(data[:, -1], dtype=torch.long)

        # Normalize the features to [0, 1]
        self.X /= 16.0

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.X[idx], self.targets[idx]


def load_optical_digits(batch_size=128, classes_to_select=[0,1,2,3,4,5,6,7,8,9]):
    train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra'
    test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes'

    full_train_dataset = OpticalDigitsDataset(train_url)
    full_test_dataset = OpticalDigitsDataset(test_url)

    train_dataset = FilteredDataset(full_train_dataset, classes_to_select)
    test_dataset  = FilteredDataset(full_test_dataset, classes_to_select)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_cifar_subset(batch_size=128,
                      classes_to_select=list(range(10)),
                      num_samples_per_class=1000):
    """
    Loads the CIFAR-10 dataset (train and test), but only includes the classes
    specified in classes_to_select. For training, only the first num_samples_per_class
    examples per (new) class are used.
    
    Args:
        batch_size (int): The batch size for the DataLoader.
        classes_to_select (list): List of original CIFAR-10 classes to select.
            (e.g., [0, 2, 4] to keep only these classes)
        num_samples_per_class (int): Number of training samples to select per class.
    
    Returns:
        train_loader, test_loader: DataLoaders for the filtered training and full test sets.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download (if needed) the full CIFAR-10 datasets.
    full_train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    full_test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    
    # Filter the datasets to keep only the desired classes.
    train_dataset = FilteredDataset(full_train_dataset, classes_to_select)
    test_dataset  = FilteredDataset(full_test_dataset, classes_to_select)
    
    # Build class-wise indices for the training set using the new (remapped) labels.
    class_indices = {new_label: [] for new_label in range(len(classes_to_select))}
    for i, (_, label) in enumerate(train_dataset):
        class_indices[label].append(i)
    
    # Select the first num_samples_per_class indices from each class.
    selected_indices = []
    for c in range(len(classes_to_select)):
        selected_indices.extend(class_indices[c][:num_samples_per_class])
    
    # Create a subset for the training set.
    train_subset = Subset(train_dataset, selected_indices)
    
    # Create DataLoaders.
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def load_fashion_mnist_subset(batch_size=128, classes_to_select=[0,1,2,3,4,5,6,7,8,9], num_samples_per_class=1000):
    """
    Loads the Fashion MNIST dataset (train and test), but only includes
    the classes specified in classes_to_select.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    
    # Download the full Fashion MNIST datasets
    full_train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    full_test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_dataset = FilteredDataset(full_train_dataset, classes_to_select)
    test_dataset  = FilteredDataset(full_test_dataset, classes_to_select)

    # Get class-wise indices for training set (ensuring the filtered dataset is indexed correctly)
    class_indices = {new_label: [] for new_label in range(len(classes_to_select))}
    
    for i, (_, label) in enumerate(train_dataset):
        class_indices[label].append(i)  # Use remapped labels from FilteredDataset

    # Select **first** `num_samples_per_class` samples from each class
    selected_indices = []
    for c in range(len(classes_to_select)):  # Work with remapped labels (0,1,...)
        selected_indices.extend(class_indices[c][:num_samples_per_class])  # Take the first `num_samples_per_class`

    # Create a fixed Subset for training
    train_subset = Subset(train_dataset, selected_indices)

    # Create DataLoaders for the subsets
    train_loader = DataLoader(
        dataset=train_subset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader
