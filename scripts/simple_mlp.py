import torch.nn as nn
import os
import torch
import math
from nn_bipartite_randomizations import BipartiteRandomization
from nn_bipartite_randomizations_sparse import BipartiteRandomizationSparse

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, activation_fn='relu'):
        super(SimpleMLP, self).__init__()
        self.input_size = input_size
        
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            if activation_fn == 'relu':
                layers.append(nn.ReLU())
            elif activation_fn == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation_fn == 'tanh':
                layers.append(nn.Tanh())
            in_size = hidden_size

        layers.append(nn.Linear(in_size, num_classes))
        layers.append(nn.LogSoftmax(dim=1))        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_size)  # Flatten the image
        return self.network(x)

    def permuted_forward(self, x, permute_order):
        out = self.forward(x)
        out = out[:, permute_order]
        return out

    def signed_unweighted_forward(self, x):
        x = x.view(-1, self.input_size)  # Flatten the image
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                signed_weight = torch.sign(layer.weight.data.clone())
                signed_bias = torch.sign(layer.bias.data.clone()) if layer.bias is not None else None
                x = nn.functional.linear(x, signed_weight, signed_bias)
            else:
                x = layer(x)  # Forward pass through non-linear layers (e.g., Sigmoid, BatchNorm, etc.)
        return x

    def signed_unweighted_forward_first_layer(self, x):
        x = x.view(-1, self.input_size)  # Flatten the image
        first_layer = True
        for layer in self.network:
            if isinstance(layer, nn.Linear) and first_layer:
                signed_weight = torch.sign(layer.weight.data.clone())
                signed_bias = torch.sign(layer.bias.data.clone()) if layer.bias is not None else None
                x = nn.functional.linear(x, signed_weight, signed_bias)
                first_layer = False  # Only apply to the first layer
            else:
                x = layer(x)  # Forward pass through other layers (e.g., Sigmoid, BatchNorm, etc.)
        return x

    def signed_unweighted_forward_last_layer(self, x):
        x = x.view(-1, self.input_size)  # Flatten the image
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear) and i == len(self.network) - 2:  # Second-to-last layer is the last nn.Linear
                signed_weight = torch.sign(layer.weight.data.clone())
                signed_bias = torch.sign(layer.bias.data.clone()) if layer.bias is not None else None
                x = nn.functional.linear(x, signed_weight, signed_bias)
            else:
                x = layer(x)  # Forward pass through other layers (e.g., Sigmoid, BatchNorm, etc.)
        return x

    def unsigned_weighted_forward(self, x):
        x = x.view(-1, self.input_size)  # Flatten the image
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                signed_weight = torch.abs(layer.weight.data.clone())
                signed_bias = torch.abs(layer.bias.data.clone()) if layer.bias is not None else None
                x = nn.functional.linear(x, signed_weight, signed_bias)
            else:
                x = layer(x)  # Forward pass through non-linear layers (e.g., Sigmoid, BatchNorm, etc.)
        return x
    
    def threshold_weight_forward(self, x, method='smallest', fraction_non_zero=1.0, to_signs=False):
        """
        Sparsify neural network weights and biases using different methods:
        1. 'smallest': Retain the specified fraction of weights with the largest absolute values.
        2. 'highest': Retain the specified fraction of weights with the smallest absolute values.
        3. 'random': Retain a specified fraction of weights as non-zero, randomly.
        
        Args:
            x: Input tensor.
            method: The sparsification method ('smallest', 'highest', or 'random').
            fraction_non_zero: Fraction of weights and biases to retain as non-zero.
        
        Returns:
            x: Output tensor after forward pass with sparsified weights and biases.
            fraction_of_non_zeros: List of fractions of non-zero weights per layer.
        """
        fraction_of_non_zeros = []
        x = x.view(-1, self.input_size)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data.clone()
                bias = layer.bias.data.clone()
                
                total_weights = weight.numel()
                num_non_zero = int(fraction_non_zero * total_weights)
                total_biases = bias.numel()
                num_non_zero_bias = int(fraction_non_zero * total_biases)
                
                if method in ['smallest', 'highest']:
                    # Sparsify weights
                    abs_weight = torch.abs(weight).view(-1)
                    if method == 'smallest':
                        # Retain the largest absolute values
                        _, indices = torch.topk(abs_weight, num_non_zero, largest=True)
                    elif method == 'highest':
                        # Retain the smallest absolute values
                        _, indices = torch.topk(abs_weight, num_non_zero, largest=False)
                    
                    mask = torch.zeros_like(abs_weight)
                    mask[indices] = 1
                    weight = weight.view(-1) * mask
                    weight = weight.view_as(layer.weight)
                    
                    # Sparsify biases
                    abs_bias = torch.abs(bias)
                    if method == 'smallest':
                        _, indices_bias = torch.topk(abs_bias, num_non_zero_bias, largest=True)
                    elif method == 'highest':
                        _, indices_bias = torch.topk(abs_bias, num_non_zero_bias, largest=False)
                    
                    mask_bias = torch.zeros_like(bias)
                    mask_bias[indices_bias] = 1
                    bias = bias * mask_bias
                
                elif method == 'random':
                    # Randomly retain weights
                    mask = torch.zeros(total_weights)
                    mask[:num_non_zero] = 1
                    mask = mask[torch.randperm(total_weights)]
                    weight = weight.view(-1) * mask
                    weight = weight.view_as(layer.weight)
                    
                    # Randomly retain biases
                    mask_bias = torch.zeros(total_biases)
                    mask_bias[:num_non_zero_bias] = 1
                    mask_bias = mask_bias[torch.randperm(total_biases)]
                    bias = bias * mask_bias
                
                else:
                    raise ValueError("Invalid method. Use 'smallest', 'highest', or 'random'.")
                
                # Convert to only signs
                if to_signs:
                    weight = torch.sign(weight)
                    bias = torch.sign(bias)

                # Calculate fraction of non-zero weights
                fraction_non_zero_actual = weight.nonzero().size(0) / weight.numel()
                fraction_of_non_zeros.append(fraction_non_zero_actual)
                
                x = nn.functional.linear(x, weight, bias)
            else:
                # Forward pass through other layers
                x = layer(x)
        
        return x, fraction_of_non_zeros

    def flip_weights_forward(self, x, q=0.1, method='smallest', to_signs=False):
        """
        Flip the sign of a q fraction of weights and biases in each nn.Linear layer, based on magnitude.

        The selection of q fraction is controlled by the `method` parameter:
        - 'smallest': Flip the sign of the smallest q fraction by magnitude.
        - 'largest': Flip the sign of the largest q fraction by magnitude.
        - 'random': Randomly select q fraction of weights to flip.

        After flipping, if `to_signs` is True, convert all weights and biases to their sign (-1 or 1).

        Args:
            x (torch.Tensor): Input tensor.
            q (float): Fraction of elements to flip (0 < q <= 1).
            method (str): 'smallest', 'largest', or 'random'.
            to_signs (bool): Convert outputs to signs if True.

        Returns:
            torch.Tensor: Output tensor after modified forward pass.
        """
        x = x.view(-1, self.input_size)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                # Clone weights and biases
                weight = layer.weight.data.clone()
                bias = layer.bias.data.clone()

                # Flatten
                weight_flat = weight.view(-1)
                total_weights = weight_flat.numel()
                num_to_flip = int(q * total_weights)

                # Select indices by magnitude
                if method == 'smallest':
                    abs_vals = weight_flat.abs()
                    _, sorted_indices = torch.sort(abs_vals, descending=False)
                    flip_indices = sorted_indices[:num_to_flip]
                elif method == 'largest':
                    abs_vals = weight_flat.abs()
                    _, sorted_indices = torch.sort(abs_vals, descending=True)
                    flip_indices = sorted_indices[:num_to_flip]
                elif method == 'random':
                    perm = torch.randperm(total_weights, device=weight.device)
                    flip_indices = perm[:num_to_flip]
                else:
                    raise ValueError("Invalid method. Use 'smallest', 'largest', or 'random'.")

                # Flip selected weights
                weight_flat[flip_indices] = -weight_flat[flip_indices]
                weight = weight_flat.view_as(weight)

                # Process biases similarly
                bias_flat = bias.view(-1)
                total_biases = bias_flat.numel()
                num_bias_to_flip = int(q * total_biases)

                if method == 'smallest':
                    abs_bias = bias_flat.abs()
                    _, sorted_bias_indices = torch.sort(abs_bias, descending=False)
                    flip_bias_indices = sorted_bias_indices[:num_bias_to_flip]
                elif method == 'largest':
                    abs_bias = bias_flat.abs()
                    _, sorted_bias_indices = torch.sort(abs_bias, descending=True)
                    flip_bias_indices = sorted_bias_indices[:num_bias_to_flip]
                elif method == 'random':
                    perm_b = torch.randperm(total_biases, device=bias.device)
                    flip_bias_indices = perm_b[:num_bias_to_flip]
                else:
                    raise ValueError("Invalid method. Use 'smallest', 'largest', or 'random'.")

                bias_flat[flip_bias_indices] = -bias_flat[flip_bias_indices]
                bias = bias_flat.view_as(bias)

                # Convert to sign if requested
                if to_signs:
                    weight = torch.sign(weight)
                    bias = torch.sign(bias)

                # Apply modified layer
                x = nn.functional.linear(x, weight, bias)
            else:
                # Pass through other layers
                x = layer(x)
        return x

    def forward_with_noise(self, x, sigma=0.0, to_signs=False):
        """
        Forward pass that injects Gaussian noise into the weights of every nn.Linear layer,
        but only for weights and biases that are nonzero.
        Optionally converts the weights and biases to their sign.

        Parameters:
            x: Input tensor.
            sigma: Standard deviation of the Gaussian noise.
            to_signs: If True, convert weights and biases to {-1, 0, 1} using torch.sign.
        """
        x = x.view(-1, self.input_size)  # Flatten the image
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data.clone()
                bias = layer.bias.data.clone()
                
                # Add Gaussian noise if sigma > 0, but only for nonzero weights and biases
                if sigma > 0:
                    # Create masks for nonzero elements
                    weight_mask = (weight != 0).float()
                    bias_mask = (bias != 0).float()
                    
                    # Generate noise and apply only where the mask is 1
                    weight = weight + torch.randn_like(weight) * sigma * weight_mask
                    bias = bias + torch.randn_like(bias) * sigma * bias_mask
                
                # Optionally convert to signs
                if to_signs:
                    weight = torch.sign(weight)
                    bias = torch.sign(bias)
                
                # Perform the linear operation with the modified weights
                x = nn.functional.linear(x, weight, bias)
            else:
                # For non-linear layers (e.g., ReLU, LogSoftmax), apply as usual
                x = layer(x)
        return x
    

    def forward_with_uniform_noise(self, x, a=0.0, to_signs=False, is_multiplicative=False):
        """
        Forward pass that injects Uniform(−a, a) noise into every nn.Linear layer’s
        nonzero weights and biases. Optionally converts them to {-1, 0, +1}.

        Parameters:
            x:       Input tensor.
            a:       Half-width of the uniform distribution.
            to_signs: If True, convert weights/biases via torch.sign after adding noise.
        """
        x = x.view(-1, self.input_size)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data.clone()
                bias   = layer.bias.data.clone()

                if a > 0:
                    # masks for nonzero params
                    w_mask = (weight != 0).float()
                    b_mask = (bias   != 0).float()

                    # U(−a, a) = (rand()*2a − a)
                    noise_w = (torch.rand_like(weight) * 2 * a - a) * w_mask
                    noise_b = (torch.rand_like(bias)   * 2 * a - a) * b_mask

                    if is_multiplicative:
                        weight = weight * noise_w
                        bias   = bias    * noise_b
                    else:
                        weight = weight + noise_w
                        bias   = bias   + noise_b

                if to_signs:
                    weight = torch.sign(weight)
                    bias   = torch.sign(bias)

                x = nn.functional.linear(x, weight, bias)
            else:
                x = layer(x)
        return x


    def modify_weights_in_place_with_noise(self, sigma=0.0):
        """
        Modifies the weights and biases of the network in-place by adding Gaussian noise
        to non-zero values only. Weights and biases with a value of zero remain unchanged.

        Args:
            sigma (float): Standard deviation of the Gaussian noise.
        """
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                # Process weights:
                weight = layer.weight.data  # in-place modification
                # Create a mask that is 1 where weight != 0 and 0 otherwise.
                non_zero_mask = (weight != 0).float()
                # Generate noise of the same shape as the weights.
                noise = torch.randn_like(weight) * sigma
                # Only add noise to non-zero weights.
                weight.add_(noise * non_zero_mask)

                # Process biases:
                bias = layer.bias.data
                non_zero_bias_mask = (bias != 0).float()
                noise_bias = torch.randn_like(bias) * sigma
                bias.add_(noise_bias * non_zero_bias_mask)


    def modify_weights_in_place(self, method='smallest', fraction_non_zero=1.0):
        """
        Modifies the weights and biases of the network in-place using a specified sparsification method.

        Args:
            method (str): The sparsification method ('smallest', 'highest', or 'random').
            fraction_non_zero (float): Fraction of weights and biases to retain as non-zero.

        Returns:
            fraction_of_non_zeros (list): List of fractions of non-zero weights per layer.
        """
        fraction_of_non_zeros = []
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                # Clone weights and biases for manipulation
                weight = layer.weight.data
                bias = layer.bias.data

                # Calculate number of non-zero weights and biases
                total_weights = weight.numel()
                num_non_zero = int(fraction_non_zero * total_weights)
                total_biases = bias.numel()
                num_non_zero_bias = int(fraction_non_zero * total_biases)

                if method in ['smallest', 'highest']:
                    abs_weight = torch.abs(weight).view(-1)
                    if method == 'smallest':
                        _, indices = torch.topk(abs_weight, num_non_zero, largest=True)
                    elif method == 'highest':
                        _, indices = torch.topk(abs_weight, num_non_zero, largest=False)

                    # Create mask and sparsify weights
                    mask = torch.zeros_like(abs_weight)
                    mask[indices] = 1
                    weight.view(-1).mul_(mask)

                    # Sparsify biases
                    abs_bias = torch.abs(bias)
                    if method == 'smallest':
                        _, indices_bias = torch.topk(abs_bias, num_non_zero_bias, largest=True)
                    elif method == 'highest':
                        _, indices_bias = torch.topk(abs_bias, num_non_zero_bias, largest=False)

                    mask_bias = torch.zeros_like(bias)
                    mask_bias[indices_bias] = 1
                    bias.mul_(mask_bias)

                elif method == 'random':
                    # Randomly sparsify weights
                    mask = torch.zeros(total_weights)
                    mask[:num_non_zero] = 1
                    mask = mask[torch.randperm(total_weights)]
                    weight.view(-1).mul_(mask)

                    # Randomly sparsify biases
                    mask_bias = torch.zeros(total_biases)
                    mask_bias[:num_non_zero_bias] = 1
                    mask_bias = mask_bias[torch.randperm(total_biases)]
                    bias.mul_(mask_bias)

                else:
                    raise ValueError("Invalid method. Use 'smallest', 'highest', or 'random'.")

                # Calculate fraction of non-zero weights
                fraction_non_zero_actual = weight.nonzero().size(0) / total_weights
                fraction_of_non_zeros.append(fraction_non_zero_actual)

        return fraction_of_non_zeros


    def extract_weights_as_layered_edgelists(self):
        """
        Extracts the weights of each layer of the MLP as a directed, weighted graph
        in edgelist format per layer.
        """
        layer_edgelists = {}
        neuron_idx = 0
        for idx, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                edgelist = []
                weight_matrix = layer.weight.detach().cpu().numpy()
                for i in range(weight_matrix.shape[0]):
                    for j in range(weight_matrix.shape[1]):
                        #if weight_matrix[i, j] != 0:
                        input_neuron = f"L{neuron_idx}_{j}"
                        output_neuron = f"L{neuron_idx + 1}_{i}"
                        edgelist.append(
                            (input_neuron, output_neuron, weight_matrix[i, j])
                        )
                layer_edgelists[f"Layer{neuron_idx}"] = edgelist
                neuron_idx += 1
        return layer_edgelists

    def save_edgelists_to_files(self, epoch, directory=".", add_network_sizes=False, suffix=''):
        """
        Saves the edgelists for each layer to a file, with one file per layer.

        Args:
            epoch (int): The epoch number to include in the filename.
            directory (str): The directory to save the files in.
        """
        os.makedirs(directory, exist_ok=True)
        edgelists = self.extract_weights_as_layered_edgelists()
        # Save each edgelist to a separate file
        for layer, edgelist in edgelists.items():
            filename = f"{directory}/epoch_{epoch}_{layer}_edgelist{suffix}.txt"
            if add_network_sizes:
                # Get the unique nodes in the first and second columns
                nodes_in_col1 = set(edge[0] for edge in edgelist)
                nodes_in_col2 = set(edge[1] for edge in edgelist)
                num_unique_col1 = len(nodes_in_col1)
                num_unique_col2 = len(nodes_in_col2)
                
            with open(filename, "w") as f:
                if add_network_sizes:
                    f.write(f"{num_unique_col1} {num_unique_col2}\n")
                for edge in edgelist:
                    f.write(f"{edge[0]} {edge[1]} {edge[2]:.6f}\n")


    def save_all_edgelists_to_edgelist(self, epoch, directory="."):
        os.makedirs(directory, exist_ok=True)
        edgelists = self.extract_weights_as_layered_edgelists()
        flattened_edgelists = [x for xs in edgelists.values() for x in xs]

        filename = f"{directory}/epoch_{epoch}_all_edgelists.txt"
        with open(filename, "w") as f:
            for edge in flattened_edgelists:
                f.write(f"{edge[0]} {edge[1]} {edge[2]:.6f}\n")


    def count_parameters(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)


    def load_layer_weights_from_file(self, filename, layer_idx, shuffle_type='', num_swaps=None, is_sparse=False):
        """
        Load weights from an edgelist file and set them to a specific layer in the model.
        
        Args:
            filename (str): Path to the edgelist file.
            layer_idx (int): The index of the Linear layer in the model to set the weights.
        """
        input_nodes = {}
        output_nodes = {}
        edges = []
        weights = []
        with open(filename, "r") as f:
            f.readline()
            for line in f:
                node1, node2, weight = line.strip().split()
                weight = float(weight)
                if node1 not in input_nodes:
                    input_nodes[node1] = len(input_nodes)
                if node2 not in output_nodes:
                    output_nodes[node2] = len(output_nodes)
                edges.append((input_nodes[node1], output_nodes[node2], weight))
                weights.append(weight)

        fraction_rewired_links = None

        if shuffle_type == 'random':
            if is_sparse:
                edges = BipartiteRandomizationSparse.shuffle_weights(edges)
            else:
                edges = BipartiteRandomization.shuffle_weights(edges)
        elif shuffle_type == 'random_l_links':
            edges = BipartiteRandomization.shuffle_l_weights(edges, L=num_swaps)
        elif shuffle_type == 'random_pos_neg':
            if is_sparse:
                edges = BipartiteRandomizationSparse.randomize_edges_with_sign_separation(edges)
            else:
                edges = BipartiteRandomization.randomize_edges_with_sign_separation(edges)
        elif shuffle_type == 'random_pos_neg_fraction':
            edges = BipartiteRandomization.randomize_edges_with_sign_separation_fraction_rewire(edges, num_swaps=num_swaps)
        elif shuffle_type == 'keep_total_strength_left':
            if is_sparse:
                edges = BipartiteRandomizationSparse.reshuffle_all_weights_left(edges)
            else:
                edges = BipartiteRandomization.reshuffle_all_weights_left(edges)
        elif shuffle_type == 'keep_total_strength_right':
            if is_sparse:
                edges = BipartiteRandomizationSparse.reshuffle_all_weights_right(edges)
            else:
                edges = BipartiteRandomization.reshuffle_all_weights_right(edges)
        elif shuffle_type == 'keep_pos_neg_strength_left':
            if is_sparse:            
                edges = BipartiteRandomizationSparse.reshuffle_separate_signs_left(edges)
            else:
                edges = BipartiteRandomization.reshuffle_separate_signs_left(edges)
        elif shuffle_type == 'keep_pos_neg_strength_right':
            if is_sparse:
                edges = BipartiteRandomizationSparse.reshuffle_separate_signs_right(edges)
            else:
                edges = BipartiteRandomization.reshuffle_separate_signs_right(edges)
        elif shuffle_type == 'keep_in_out_degree_swap_edges':
            if is_sparse:
                edges, fraction_rewired_links = BipartiteRandomizationSparse.positive_edge_switching_algorithm(edges, num_swaps=num_swaps)
            else:
                edges, fraction_rewired_links = BipartiteRandomization.positive_edge_switching_algorithm(edges, num_swaps=num_swaps)

        input_size = len(input_nodes)
        output_size = len(output_nodes)
        
        weight_matrix = torch.zeros((output_size, input_size), dtype=torch.float32)
        
        for in_idx, out_idx, weight in edges:
            weight_matrix[out_idx, in_idx] = weight
        
        layer = self.network[layer_idx]
        if isinstance(layer, nn.Linear):
            layer.weight.data = weight_matrix

        return fraction_rewired_links
    

    def shuffle_model_weights(self, shuffle_type, num_swaps=None, is_sparse=False):
        """
        Shuffles the weights of the model using an edge-based randomization method.
        
        This function converts each nn.Linear layer's weight matrix into an edgelist,
        applies a specified shuffling routine (e.g., random, random_l_links, random_pos_neg, etc.)
        using the corresponding BipartiteRandomization function, and then reconstructs the weight matrix.
        
        Args:
            shuffle_type (str): The type of shuffling to apply. Options are:
                'random', 'random_l_links', 'random_pos_neg', 'random_pos_neg_fraction',
                'keep_total_strength_left', 'keep_total_strength_right',
                'keep_pos_neg_strength_left', 'keep_pos_neg_strength_right',
                'keep_in_out_degree_swap_edges'.
            num_swaps (int, optional): If applicable for the chosen shuffling method, the number of swap operations.
            is_sparse (bool, optional): Whether to use the sparse version of the randomization.
            
        Returns:
            fraction_rewired_links_all (list or None): If the shuffling method returns a fraction of rewired links,
                a list with that value per layer is returned. Otherwise, returns None.
        """
        fraction_rewired_links_all = []
        
        # Loop over all layers in the network
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                # Extract current weight matrix
                weight_matrix = layer.weight.data.clone()
                output_size, input_size = weight_matrix.shape

                # Convert the weight matrix to a list of edges.
                # Each edge is a tuple: (input_index, output_index, weight_value)
                edgelist = []
                for out_idx in range(output_size):
                    for in_idx in range(input_size):
                        edgelist.append((in_idx, out_idx, weight_matrix[out_idx, in_idx].item()))
                
                # Apply the desired shuffling based on the shuffle_type argument.
                if shuffle_type == 'random':
                    if is_sparse:
                        shuffled_edges = BipartiteRandomizationSparse.shuffle_weights(edgelist)
                    else:
                        shuffled_edges = BipartiteRandomization.shuffle_weights(edgelist)
                elif shuffle_type == 'random_l_links':
                    shuffled_edges = BipartiteRandomization.shuffle_l_weights(edgelist, L=num_swaps)
                elif shuffle_type == 'random_pos_neg':
                    if is_sparse:
                        shuffled_edges = BipartiteRandomizationSparse.randomize_edges_with_sign_separation(edgelist)
                    else:
                        shuffled_edges = BipartiteRandomization.randomize_edges_with_sign_separation(edgelist)
                elif shuffle_type == 'random_pos_neg_fraction':
                    shuffled_edges = BipartiteRandomization.randomize_edges_with_sign_separation_fraction_rewire(edgelist, num_swaps=num_swaps)
                elif shuffle_type == 'keep_total_strength_left':
                    if is_sparse:
                        shuffled_edges = BipartiteRandomizationSparse.reshuffle_all_weights_left(edgelist)
                    else:
                        shuffled_edges = BipartiteRandomization.reshuffle_all_weights_left(edgelist)
                elif shuffle_type == 'keep_total_strength_right':
                    if is_sparse:
                        shuffled_edges = BipartiteRandomizationSparse.reshuffle_all_weights_right(edgelist)
                    else:
                        shuffled_edges = BipartiteRandomization.reshuffle_all_weights_right(edgelist)
                elif shuffle_type == 'keep_pos_neg_strength_left':
                    if is_sparse:
                        shuffled_edges = BipartiteRandomizationSparse.reshuffle_separate_signs_left(edgelist)
                    else:
                        shuffled_edges = BipartiteRandomization.reshuffle_separate_signs_left(edgelist)
                elif shuffle_type == 'keep_pos_neg_strength_right':
                    if is_sparse:
                        shuffled_edges = BipartiteRandomizationSparse.reshuffle_separate_signs_right(edgelist)
                    else:
                        shuffled_edges = BipartiteRandomization.reshuffle_separate_signs_right(edgelist)
                elif shuffle_type == 'keep_in_out_degree_swap_edges':
                    if is_sparse:
                        shuffled_edges, fraction_rewired_links = BipartiteRandomizationSparse.positive_edge_switching_algorithm(edgelist, num_swaps=num_swaps)
                    else:
                        shuffled_edges, fraction_rewired_links = BipartiteRandomization.positive_edge_switching_algorithm(edgelist, num_swaps=num_swaps)
                    fraction_rewired_links_all.append(fraction_rewired_links)
                else:
                    raise ValueError("Invalid shuffle type. Use one of the recognized methods.")
                
                # Create a new weight matrix from the shuffled edges.
                new_weight_matrix = torch.zeros_like(weight_matrix)
                for in_idx, out_idx, weight in shuffled_edges:
                    new_weight_matrix[out_idx, in_idx] = weight
                
                # Update the layer's weights with the shuffled matrix.
                layer.weight.data = new_weight_matrix
        
        if fraction_rewired_links_all:
            return fraction_rewired_links_all
        else:
            return None
    

    def input_pixel_activity_map(self,
                                 method: str = 'smallest',
                                 fraction_non_zero: float = 1.0):
        """
        Compute per-input-pixel activity after sparsifying the FIRST layer.

        Args:
            method:            'smallest' (keep largest-|w|), 
                               'highest' (keep smallest-|w|), 
                               'random'
            fraction_non_zero: fraction of weights to RETAIN (0–1)

        Returns:
            activity_sum:   Tensor (H, W) of sum |w_ij| over hidden units j
            activity_count: Tensor (H, W) of count non-zero weights per pixel
        """
        # 1) locate & clone first linear layer
        first_lin = next(l for l in self.network if isinstance(l, nn.Linear))
        W = first_lin.weight.data.clone()            # shape (num_hidden, input_size)
        num_hidden, input_size = W.shape

        # 2) figure out image dims
        side = int(math.sqrt(self.input_size))
        if side * side != self.input_size:
            raise ValueError(f"input_size={self.input_size} is not a perfect square")
        H = W_img = side

        # 3) flatten & build sparsity mask exactly as in threshold_weight_forward
        flat = W.view(-1)
        total = flat.numel()
        k = int(fraction_non_zero * total)

        if method in ('smallest', 'highest'):
            abs_flat = flat.abs()
            largest = (method == 'smallest')
            _, idx = torch.topk(abs_flat, k, largest=largest)
            mask = torch.zeros_like(flat)
            mask[idx] = 1
            flat = flat * mask

        elif method == 'random':
            mask = torch.zeros(total, device=flat.device)
            mask[:k] = 1
            mask = mask[torch.randperm(total, device=flat.device)]
            flat = flat * mask

        else:
            raise ValueError("method must be 'smallest', 'highest', or 'random'")

        # 4) reshape back to (num_hidden, input_size)
        W_sparse = flat.view(num_hidden, input_size)

        # 5) aggregate per‐pixel
        #    sum of absolute weights across hidden units:
        activity_sum   = W_sparse.abs().sum(dim=0)     # shape (input_size,)
        # activity_sum   = W_sparse.sum(dim=0)     # shape (input_size,)
        #    count of non-zero connections:
        activity_count = (W_sparse != 0).sum(dim=0)    # shape (input_size,)

        # 6) reshape into (H, W)
        activity_sum   = activity_sum.view(H, W_img)
        activity_count = activity_count.view(H, W_img)

        return activity_sum, activity_count