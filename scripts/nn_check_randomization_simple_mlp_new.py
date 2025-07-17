import sys
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
            elif forward_method == 'signed':
                outputs = model.signed_unweighted_forward(images)
            else:
                raise ValueError(f"Unknown forward method: {forward_method}")

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    if verbose:
        print(f"Accuracy using {forward_method} forward method: {accuracy:.2f}%")
    
    return accuracy


def randomize_mlp(
    epoch,
    base_folder,
    classes_to_select=[0,1,2,3,4,5,6,7,8,9],
    num_samples_per_class=100,
    hidden_sizes=[128],
    dataset='MNIST',
    device="cpu",
    verbose=True,
    sparsification_levels=[0, 0.3, 0.6, 0.9] # 0 - original network
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
    
    randomization_types = ['random', 'random_pos_neg',
                           'keep_total_strength_left', 'keep_total_strength_right',
                           'keep_pos_neg_strength_left', 'keep_pos_neg_strength_right',
                           'keep_in_out_degree_swap_edges']
    # randomization_types = ['random_pos_neg']

    original_accuracy = {}
    original_signed_accuracy = {}
    random_accuracy = {}
    random_signed_accuracy = {}

    for sparse_level in sparsification_levels:
        fraction_non_zero = 1 - sparse_level

        model.load_layer_weights_from_file(f"{base_folder}/epoch_{epoch}_Layer0_edgelist.txt", layer_idx=0)
        model.load_layer_weights_from_file(f"{base_folder}/epoch_{epoch}_Layer1_edgelist.txt", layer_idx=2)

        original_accuracy[sparse_level] = compute_accuracy(model, test_loader, device, forward_method="threshold",
                                                           threshold_method='smallest', fraction_non_zero=fraction_non_zero, 
                                                           verbose=verbose, to_signs=False)
        
        original_signed_accuracy[sparse_level] = compute_accuracy(model, test_loader, device, forward_method="threshold", 
                                                                  threshold_method='smallest', fraction_non_zero=fraction_non_zero, 
                                                                  to_signs=True, verbose=verbose)

        # Randomize both layers
        acc_per_randomization = {}
        acc_signed_per_randomization = {}
        for r in randomization_types:
            model.load_layer_weights_from_file(f"{base_folder}/epoch_{epoch}_Layer0_edgelist.txt", layer_idx=0, is_sparse=True)
            model.load_layer_weights_from_file(f"{base_folder}/epoch_{epoch}_Layer1_edgelist.txt", layer_idx=2, is_sparse=True)
            
            # Sparsify first
            model.modify_weights_in_place('smallest', fraction_non_zero)
            # Randomize
            model.shuffle_model_weights(shuffle_type=r, is_sparse=True)
            # and measure accuracy
            acc_per_randomization[r] = compute_accuracy(model, test_loader, device, forward_method="base", verbose=verbose)
            acc_signed_per_randomization[r] = compute_accuracy(model, test_loader, device, forward_method='signed', verbose=verbose)
            
        random_accuracy[sparse_level] = acc_per_randomization
        random_signed_accuracy[sparse_level] = acc_signed_per_randomization

    return original_accuracy, original_signed_accuracy, random_accuracy, random_signed_accuracy