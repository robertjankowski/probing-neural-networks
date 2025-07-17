from data_utils import *
from simple_mlp import *
from tqdm import tqdm

def compute_accuracy(model, data_loader, device, forward_method="base", threshold_method='smallest', sigma=0, fraction_non_zero=1.0, to_signs=False, verbose=True):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            if forward_method == "base":
                outputs = model(images)
            elif forward_method == 'threshold':
                outputs, _ = model.threshold_weight_forward(images, method=threshold_method, fraction_non_zero=fraction_non_zero, to_signs=to_signs)
            elif forward_method == 'noise':
                outputs = model.forward_with_noise(images, sigma=sigma, to_signs=to_signs)
            else:
                raise ValueError(f"Unknown forward method: {forward_method}")

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    if verbose:
        print(f"Accuracy using {forward_method} forward method: {accuracy:.2f}%")
    
    return accuracy


def sparsify_prune_mlp(
    epoch,
    base_folder,
    classes_to_select=[0,1,2,3,4,5,6,7,8,9],
    num_samples_per_class=100,
    hidden_sizes=[128],
    dataset='MNIST',
    device="cpu",
    verbose=True,
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

    # all_fraction_non_zero = [1, 0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.47, 0.45, 0.4, 0.35, 0.3, 0.25, 0.23, 0.21, 0.2, 
    #                          0.19, 0.18, 0.17, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 
    #                          0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]
    all_fraction_non_zero = [1, 0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 
                             0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]
    
    sigmas = np.logspace(-10, 0, base=2, num=30)

    # First apply noise and later sparsify network
    results_noise_and_prune = {} 
    for sigma in tqdm(sigmas):
        model.load_layer_weights_from_file(f"{base_folder}/epoch_{epoch}_Layer0_edgelist.txt", layer_idx=0)
        model.load_layer_weights_from_file(f"{base_folder}/epoch_{epoch}_Layer1_edgelist.txt", layer_idx=2)

        # Apply noise
        model.modify_weights_in_place_with_noise(sigma)
        for fraction_non_zero in tqdm(all_fraction_non_zero):
            acc_smallest = compute_accuracy(model, test_loader, device, forward_method="threshold", 
                                            threshold_method='smallest', fraction_non_zero=fraction_non_zero, verbose=verbose)
            acc_smallest_sign = compute_accuracy(model, test_loader, device, forward_method="threshold", 
                                                threshold_method='smallest', fraction_non_zero=fraction_non_zero, to_signs=True, verbose=verbose)
            
            results_noise_and_prune[(sigma, fraction_non_zero)] = (acc_smallest, acc_smallest_sign)

    # First sparsify and later apply noise
    results_prune_and_noise = {}
    for fraction_non_zero in tqdm(all_fraction_non_zero):
        model.load_layer_weights_from_file(f"{base_folder}/epoch_{epoch}_Layer0_edgelist.txt", layer_idx=0)
        model.load_layer_weights_from_file(f"{base_folder}/epoch_{epoch}_Layer1_edgelist.txt", layer_idx=2)

        # Remove by smallest magnitute
        model.modify_weights_in_place('smallest', fraction_non_zero)
        for sigma in tqdm(sigmas):
            acc_smallest = compute_accuracy(model, test_loader, device, forward_method="noise", sigma=sigma, verbose=verbose)
            acc_smallest_sign = compute_accuracy(model, test_loader, device, forward_method="noise", sigma=sigma, to_signs=True, verbose=verbose)
        
            results_prune_and_noise[(sigma, fraction_non_zero)] = (acc_smallest, acc_smallest_sign)
 
        # TODO: add random case
        
    return original_accuracy, results_noise_and_prune, results_prune_and_noise