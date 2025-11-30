from data_utils import *
from simple_mlp import *
from tqdm import tqdm

def compute_accuracy(model, data_loader, device, forward_method="flip", flip_method='smallest', q=0, to_signs=False, verbose=True):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            if forward_method == "base":
                outputs = model(images)
            elif forward_method == 'flip':
                outputs = model.flip_weights_forward(images, q=q, method=flip_method, to_signs=to_signs)
            else:
                raise ValueError(f"Unknown forward method: {forward_method}")

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    if verbose:
        print(f"Accuracy using {forward_method} forward method: {accuracy:.2f}%")
    
    return accuracy


def flip_weights_mlp(
    epoch,
    base_folder,
    classes_to_select=[0,1,2,3,4,5,6,7,8,9],
    num_samples_per_class=100,
    hidden_sizes=[128],
    dataset='MNIST',
    device="cpu",
    verbose=True,
    activation_fn='relu'
):
    
    if dataset == 'MNIST':
        input_size = 28 * 28 * 1
        _, test_loader = load_mnist_subset(
            batch_size=128, 
            classes_to_select=classes_to_select,
            num_samples_per_class=num_samples_per_class
        )
    elif dataset == 'Fashion-MNIST':
        input_size = 28 * 28 * 1
        _, test_loader = load_fashion_mnist_subset(
            batch_size=128, 
            classes_to_select=classes_to_select,
            num_samples_per_class=num_samples_per_class
        )
    
    num_classes = len(classes_to_select)
    model = SimpleMLP(input_size, hidden_sizes, num_classes, activation_fn=activation_fn).to(device)
    
    model.load_layer_weights_from_file(f"{base_folder}/epoch_{epoch}_Layer0_edgelist.txt", layer_idx=0)
    model.load_layer_weights_from_file(f"{base_folder}/epoch_{epoch}_Layer1_edgelist.txt", layer_idx=2)
    
    # model.eval()
    original_accuracy = compute_accuracy(model, test_loader, device, forward_method="base", verbose=verbose)

    # flip_fractions = np.logspace(-11, 0, base=2, num=30)
    flip_fractions = np.linspace(0, 1, num=30)
    results = {}
    for q in tqdm(flip_fractions):
        acc_smallest = compute_accuracy(model, test_loader, device, forward_method="flip", flip_method='smallest', q=q, verbose=verbose)
        acc_smallest_sign = compute_accuracy(model, test_loader, device, forward_method="flip", flip_method='smallest', q=q, to_signs=True, verbose=verbose)

        acc_largest = compute_accuracy(model, test_loader, device, forward_method="flip", flip_method='largest', q=q, verbose=verbose)
        acc_largest_sign = compute_accuracy(model, test_loader, device, forward_method="flip", flip_method='largest', q=q, to_signs=True, verbose=verbose)
        
        acc_random = compute_accuracy(model, test_loader, device, forward_method="flip", flip_method='random', q=q, verbose=verbose)
        acc_random_sign = compute_accuracy(model, test_loader, device, forward_method="flip", flip_method='random', q=q, to_signs=True, verbose=verbose)
        
        results[q] = (acc_smallest, acc_smallest_sign, acc_largest, acc_largest_sign, acc_random, acc_random_sign)

    return original_accuracy, results