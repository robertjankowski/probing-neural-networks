from data_utils import *
from simple_mlp import *
from tqdm import tqdm

def compute_accuracy(model, data_loader, device, forward_method="base", threshold_method='smallest', fraction_non_zero=1.0, to_signs=False, verbose=True):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            if forward_method == "base":
                outputs = model(images)
            elif forward_method == 'threshold':
                outputs, _ = model.threshold_weight_forward(images, method=threshold_method, fraction_non_zero=fraction_non_zero, to_signs=to_signs)
            else:
                raise ValueError(f"Unknown forward method: {forward_method}")

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    if verbose:
        print(f"Accuracy using {forward_method} forward method: {accuracy:.2f}%")
    
    return accuracy


def sparsify_mlp(
    epoch,
    base_folder,
    classes_to_select=[0,1,2,3,4,5,6,7,8,9],
    num_samples_per_class=100,
    hidden_sizes=[128],
    dataset='MNIST',
    device="cpu",
    verbose=True,
    all_fraction_non_zero=None,
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
    model = SimpleMLP(input_size, hidden_sizes, num_classes).to(device)
    
    model.load_layer_weights_from_file(f"{base_folder}/epoch_{epoch}_Layer0_edgelist.txt", layer_idx=0)
    model.load_layer_weights_from_file(f"{base_folder}/epoch_{epoch}_Layer1_edgelist.txt", layer_idx=2)
    
    # model.eval()
    original_accuracy = compute_accuracy(model, test_loader, device, forward_method="base", verbose=verbose)

    if all_fraction_non_zero is None:
        all_fraction_non_zero = [1, 0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.47, 0.45, 0.4, 0.35, 0.3, 0.25, 0.23, 0.21, 0.2, 
                                0.19, 0.18, 0.17, 0.15, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 
                                0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]

    results = {}
    for fraction_non_zero in tqdm(all_fraction_non_zero):
        acc_smallest = compute_accuracy(model, test_loader, device, forward_method="threshold", 
                                        threshold_method='smallest', fraction_non_zero=fraction_non_zero, verbose=verbose)
        acc_smallest_sign = compute_accuracy(model, test_loader, device, forward_method="threshold", 
                                             threshold_method='smallest', fraction_non_zero=fraction_non_zero, to_signs=True, verbose=verbose)

        # Note: every time the subset of random edges might be different...
        acc_random = compute_accuracy(model, test_loader, device, forward_method="threshold", 
                                        threshold_method='random', fraction_non_zero=fraction_non_zero, verbose=verbose)
        acc_random_sign = compute_accuracy(model, test_loader, device, forward_method="threshold", 
                                             threshold_method='random', fraction_non_zero=fraction_non_zero, to_signs=True, verbose=verbose)

        results[fraction_non_zero] = (acc_smallest, acc_smallest_sign, acc_random, acc_random_sign)
        
    return original_accuracy, results