from data_utils import *
import torch.optim as optim
import torch.nn as nn
import torch
import torch.optim as optim
import copy
from simple_mlp import *
from itertools import combinations
from tqdm import tqdm


def train_model_hard(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device='cpu', verbose=False):
    best_test_accuracy = 0.0
    best_test_epoch = None
    best_test_step = None

    for epoch in range(num_epochs):
        model.train()
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Evaluate test accuracy over the entire test set.
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for test_images, test_labels in test_loader:
                    test_images, test_labels = test_images.to(device), test_labels.to(device)
                    test_outputs = model(test_images)
                    _, test_predicted = torch.max(test_outputs.data, 1)
                    test_total += test_labels.size(0)
                    test_correct += (test_predicted == test_labels).sum().item()
            test_accuracy = 100 * test_correct / test_total
            model.train()
            
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_test_epoch = epoch + 1
                best_test_step = step + 1
            
            if verbose:
                print(f"[Hard] Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, Test Acc: {test_accuracy:.2f}%")
        scheduler.step()
        if verbose:
            print(f"[Hard] Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
    return best_test_accuracy, best_test_epoch, best_test_step


def train_model_easy(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, 
                     stop_test_accuracy, patience_limit=5, device='cpu'):
    tolerance = 1e-6  # Small tolerance for floating point comparisons.
    
    # Variables for tracking the best state that did not exceed the target.
    best_state = None
    best_diff = float('inf')  # Difference: (target - test_accuracy), for test_accuracy <= target.
    
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Evaluate test accuracy over the entire test set.
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for test_images, test_labels in test_loader:
                    test_images, test_labels = test_images.to(device), test_labels.to(device)
                    test_outputs = model(test_images)
                    _, test_predicted = torch.max(test_outputs.data, 1)
                    test_total += test_labels.size(0)
                    test_correct += (test_predicted == test_labels).sum().item()
            test_accuracy = 100 * test_correct / test_total
            model.train()
            
            print(f"[Easy] Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}, Test Acc: {test_accuracy:.2f}%")
            
            # Check if the current test accuracy does not exceed the target.
            if test_accuracy <= stop_test_accuracy + tolerance:
                # Reset patience counter.
                patience_counter = 0
                diff = stop_test_accuracy - test_accuracy
                # Save state if this is the closest so far (i.e. smallest difference).
                if diff < best_diff:
                    best_diff = diff
                    best_state = {
                        'epoch': epoch + 1,
                        'step': step + 1,
                        'test_accuracy': test_accuracy,
                        'model_state': copy.deepcopy(model.state_dict())
                    }
                # If we are essentially matching the target, we can continue training a bit more
                # to see if we can get even closer.
            else:
                # Test accuracy exceeded the target. Increase patience.
                patience_counter += 1
                print(f"Test accuracy above target! Patience counter: {patience_counter}/{patience_limit}")
                # If we have exceeded our patience limit, stop training.
                if patience_counter >= patience_limit:
                    print("Patience limit reached. Stopping training.")
                    if best_state is not None:
                        # Load the best state (optional: you may decide to return it without loading).
                        model.load_state_dict(best_state['model_state'])
                        return (best_state['test_accuracy'], best_state['epoch'], best_state['step'])
                    else:
                        return (test_accuracy, epoch + 1, step + 1)
        scheduler.step()
        print(f"[Easy] Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    
    # End of training loop: return the best state found.
    if best_state is not None:
        model.load_state_dict(best_state['model_state'])
        return (best_state['test_accuracy'], best_state['epoch'], best_state['step'])
    return (test_accuracy, epoch + 1, step + 1)


def run_all_digit_pairs():
    hidden_sizes = [64]
    device = 'cpu'
    learning_rate = 0.01
    num_epochs = 10
    batch_size = 128
    input_size = 28 * 28 * 1
    num_samples_per_class = 100000
    n_iterations = 10

    activation_fn = ''

    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    output_folder = '/home/rjankow/data/task_complexity/fix_accuracy/corrected_all/'

    for a, b in tqdm(combinations(digits, 2)):
        for iteration in tqdm(range(n_iterations)):
            output_folder_task = f'{output_folder}/output_mnist_classes_{a}_{b}_dim_{hidden_sizes[0]}_n_{num_samples_per_class}_i{iteration}{activation_fn}'

            classes_to_select = [a, b]
            num_classes = len(classes_to_select)

            train_loader, test_loader = load_mnist_subset(
                batch_size=batch_size,
                classes_to_select=classes_to_select,
                num_samples_per_class=num_samples_per_class
            )
            model = SimpleMLP(input_size, hidden_sizes, num_classes).to(device)
            criterion = nn.NLLLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

            print(f"Training model on the task (digits {a} vs {b})...")
            max_test_acc, max_epoch, max_step = train_model_hard(
                model,
                train_loader,
                test_loader,
                criterion,
                optimizer,
                scheduler,
                num_epochs,
                verbose=False
            )

            # 9 here is just a placeholder, it is the best epoch (and step) 
            model.save_edgelists_to_files(9, output_folder_task, add_network_sizes=True)
            print(f"Max Test Accuracy: {max_test_acc:.2f}% at epoch {max_epoch}, step {max_step}")


def run_all_dataset():
    hidden_sizes = [64]
    device = 'cpu'
    learning_rate = 0.01
    num_epochs = 10
    batch_size = 128
    input_size = 28 * 28 * 1
    num_samples_per_class = 100000
    activation_fn = ''
    n_iterations = 10

    output_folder = '/home/rjankow/data/task_complexity/fix_accuracy/corrected_new/'
    
    classes_to_select = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for i in tqdm(range(n_iterations)):
        output_folder_all = f'{output_folder}/output_mnist_classes_all_dim_{hidden_sizes[0]}_n_{num_samples_per_class}_i{i}{activation_fn}'
    
        num_classes = len(classes_to_select)

        train_loader, test_loader = load_mnist_subset(
            batch_size=batch_size,
            classes_to_select=classes_to_select,
            num_samples_per_class=num_samples_per_class
        )
        model = SimpleMLP(input_size, hidden_sizes, num_classes).to(device)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        print(f"Training model all digits...")
        max_test_acc, max_epoch, max_step = train_model_hard(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs,
            verbose=True
        )

        model.save_edgelists_to_files(9, output_folder_all, add_network_sizes=True)
        print(f"Max Test Accuracy: {max_test_acc:.2f}% at epoch {max_epoch}, step {max_step}")


def main(iteration):
    hidden_sizes = [64] # [64]
    device = 'cpu'
    learning_rate = 0.01
    num_epochs = 10
    batch_size = 128
    input_size = 28 * 28 * 1
    num_samples_per_class = 100000

    activation_fn = '_relu'
    # activation_fn = '_sigmoid'

    # output_folder = '/home/rjankow/data/task_complexity/fix_accuracy/corrected/'
    # output_folder = '/home/rjankow/data/task_complexity/fix_accuracy/corrected_new/'
    output_folder = '/home/rjankow/data/task_complexity/fix_accuracy/for_paper/'
    output_folder_hard = f'{output_folder}/output_mnist_classes_7_9_dim_{hidden_sizes[0]}_n_{num_samples_per_class}_i{iteration}{activation_fn}'
    output_folder_easy = f'{output_folder}/output_mnist_classes_0_7_dim_{hidden_sizes[0]}_n_{num_samples_per_class}_i{iteration}{activation_fn}'
    # output_folder_hard = f'{output_folder}/output_mnist_classes_1_6_dim_{hidden_sizes[0]}_n_{num_samples_per_class}_i{iteration}'
    # output_folder_easy = f'{output_folder}/output_mnist_classes_0_6_dim_{hidden_sizes[0]}_n_{num_samples_per_class}_i{iteration}'

    ###############################################################################
    # 1. TRAINING ON THE HARD TASK (digits 7 vs 9)
    ###############################################################################

    hard_classes_to_select = [7, 9]
    num_classes_hard = len(hard_classes_to_select)

    hard_train_loader, hard_test_loader = load_mnist_subset(
        batch_size=batch_size,
        classes_to_select=hard_classes_to_select,
        num_samples_per_class=num_samples_per_class
    )

    hard_model = SimpleMLP(input_size, hidden_sizes, num_classes_hard).to(device)
    hard_criterion = nn.NLLLoss()
    hard_optimizer = optim.Adam(hard_model.parameters(), lr=learning_rate)
    hard_scheduler = optim.lr_scheduler.CosineAnnealingLR(hard_optimizer, T_max=num_epochs)

    # hard_model.save_edgelists_to_files(0, output_folder_hard, add_network_sizes=True)

    print("Training model on the hard task (digits 7 vs 9)...")
    hard_max_test_acc, hard_max_epoch, hard_max_step = train_model_hard(
        hard_model,
        hard_train_loader,
        hard_test_loader,
        hard_criterion,
        hard_optimizer,
        hard_scheduler,
        num_epochs
    )

    hard_model.save_edgelists_to_files(9, output_folder_hard, add_network_sizes=True)

    print(f"\nHard Task Results:")
    print(f"Max Test Accuracy: {hard_max_test_acc:.2f}% at epoch {hard_max_epoch}, step {hard_max_step}")

    ###############################################################################
    # 2. TRAINING ON THE EASY TASK (digits 0 vs 7)
    #    We stop the easy task once its test accuracy reaches (but does not exceed)
    #    the hard task's max test accuracy. We allow a few extra iterations (patience)
    #    if the model overshoots.
    ###############################################################################

    easy_classes_to_select = [0, 7]
    num_classes_easy = len(easy_classes_to_select)

    easy_train_loader, easy_test_loader = load_mnist_subset(
        batch_size=batch_size,
        classes_to_select=easy_classes_to_select,
        num_samples_per_class=num_samples_per_class
    )

    easy_model = SimpleMLP(input_size, hidden_sizes, num_classes_easy).to(device)
    easy_criterion = nn.NLLLoss()
    easy_optimizer = optim.Adam(easy_model.parameters(), lr=learning_rate)
    easy_scheduler = optim.lr_scheduler.CosineAnnealingLR(easy_optimizer, T_max=num_epochs)

    # easy_model.save_edgelists_to_files(0, output_folder_easy, add_network_sizes=True)

    print("\nTraining model on the easy task (digits 0 vs 7)...")

    # When fixing the accuracy for the hard task
    # easy_final_test_acc, easy_final_epoch, easy_final_step = train_model_easy(
    #     easy_model,
    #     easy_train_loader,
    #     easy_test_loader,
    #     easy_criterion,
    #     easy_optimizer,
    #     easy_scheduler,
    #     num_epochs,
    #     stop_test_accuracy=100, #hard_max_test_acc,  # target from the hard task
    #     patience_limit=5  # you can adjust this number as needed
    # )

    easy_final_test_acc, easy_final_epoch, easy_final_step = train_model_hard(
        easy_model,
        easy_train_loader,
        easy_test_loader,
        easy_criterion,
        easy_optimizer,
        easy_scheduler,
        num_epochs,
    )

    # Let us assume that the best epoch is 9
    easy_model.save_edgelists_to_files(9, output_folder_easy, add_network_sizes=True)

    print(f"\nEasy Task Results:")
    print(f"Final Test Accuracy: {easy_final_test_acc:.2f}% at epoch {easy_final_epoch}, step {easy_final_step}")


# if __name__ == '__main__':
    # for i in tqdm(range(100)):
        # main(iteration=i)