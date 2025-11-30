from data_utils import *
from simple_mlp import *
from tqdm import tqdm

def compute_accuracy(model, data_loader, device, forward_method="noise", sigma=0, to_signs=False, verbose=True, is_multiplicative=False):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            if forward_method == "base":
                outputs = model(images)
            elif forward_method == 'noise':
                outputs = model.forward_with_noise(images, sigma=sigma, to_signs=to_signs)
            elif forward_method == 'noise_uniform':
                outputs = model.forward_with_uniform_noise(images, a=sigma, to_signs=to_signs, is_multiplicative=is_multiplicative)
            else:
                raise ValueError(f"Unknown forward method: {forward_method}")

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    if verbose:
        print(f"Accuracy using {forward_method} forward method: {accuracy:.2f}%")
    
    return accuracy


def gaussian_noise_mlp(
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

    noise_forward_method = 'noise_uniform'
    # noise_forward_method = 'noise' #'noise' - gaussian noise
    is_multiplicative = False

    sigmas = np.logspace(-10, 0, base=2, num=30)
    results = {}
    for sigma in tqdm(sigmas):
        acc_smallest = compute_accuracy(model, test_loader, device, forward_method=noise_forward_method, 
                                        sigma=sigma, verbose=verbose, is_multiplicative=is_multiplicative)
        acc_smallest_sign = compute_accuracy(model, test_loader, device, forward_method=noise_forward_method, 
                                             sigma=sigma, to_signs=True, verbose=verbose, is_multiplicative=is_multiplicative)
        results[sigma] = (acc_smallest, acc_smallest_sign)

    return original_accuracy, results