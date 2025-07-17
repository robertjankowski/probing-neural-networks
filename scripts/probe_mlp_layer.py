import torch
from nn_bipartite_randomizations import BipartiteRandomization
from nn_bipartite_randomizations_sparse import BipartiteRandomizationSparse


def apply_noise_to_weights(weight_matrix: torch.Tensor,
                           sigma: float = 0.0,
                           to_signs: bool = False) -> torch.Tensor:
    """
    Add Gaussian noise to non-zero weights.

    Args:
        weight_matrix: torch.Tensor of shape (out_features, in_features).
        sigma: standard deviation of Gaussian noise.
        to_signs: if True, convert weights to {-1, 0, 1}.

    Returns:
        Modified weight matrix.
    """
    w = weight_matrix.clone()
    if sigma > 0:
        mask = (w != 0).float()
        noise = torch.randn_like(w) * sigma
        w = w + noise * mask
    if to_signs:
        w = torch.sign(w)
    return w


def flip_weights(weight_matrix: torch.Tensor,
                 q: float = 0.1,
                 method: str = 'smallest',
                 to_signs: bool = False) -> torch.Tensor:
    """
    Flip the sign of a fraction q of weights based on magnitude or at random.

    Args:
        weight_matrix: torch.Tensor
        q: fraction of elements to flip (0 < q <= 1)
        method: 'smallest', 'largest', or 'random'
        to_signs: if True, convert to {-1, 0, 1}

    Returns:
        Modified weight matrix.
    """
    w = weight_matrix.clone()
    flat = w.view(-1)
    total = flat.numel()
    num = int(q * total)

    if method in ('smallest', 'largest'):
        vals = flat.abs()
        descending = (method == 'largest')
        _, idx = torch.sort(vals, descending=descending)
        flip_idx = idx[:num]
    elif method == 'random':
        perm = torch.randperm(total, device=w.device)
        flip_idx = perm[:num]
    else:
        raise ValueError("method must be 'smallest', 'largest', or 'random'")

    flat[flip_idx] = -flat[flip_idx]
    w = flat.view_as(w)

    if to_signs:
        w = torch.sign(w)
    return w


def threshold_weights(weight_matrix: torch.Tensor,
                      method: str = 'smallest',
                      fraction_non_zero: float = 1.0,
                      to_signs: bool = False) -> torch.Tensor:
    """
    Sparsify weights by retaining only a fraction of elements by magnitude or at random.

    Args:
        weight_matrix: torch.Tensor
        method: 'smallest', 'highest', or 'random'
        fraction_non_zero: fraction of weights to keep
        to_signs: if True, convert to {-1, 0, 1}

    Returns:
        Modified weight matrix.
    """
    w = weight_matrix.clone()
    flat = w.view(-1)
    total = flat.numel()
    keep = int(fraction_non_zero * total)

    if method in ('smallest', 'highest'):
        vals = flat.abs()
        largest = (method == 'smallest')
        _, idx = torch.topk(vals, keep, largest=largest)
        mask = torch.zeros_like(vals)
        mask[idx] = 1
        flat = flat * mask
    elif method == 'random':
        mask = torch.zeros(total, device=w.device)
        mask[:keep] = 1
        mask = mask[torch.randperm(total, device=w.device)]
        flat = flat * mask
    else:
        raise ValueError("method must be 'smallest', 'highest', or 'random'")

    w = flat.view_as(w)
    if to_signs:
        w = torch.sign(w)
    return w


def shuffle_weights(weight_matrix: torch.Tensor,
                    shuffle_type: str = 'random',
                    num_swaps: int = None,
                    is_sparse: bool = False) -> torch.Tensor:
    """
    Shuffle weights using bipartite randomization routines or simple permutation.

    Args:
        weight_matrix: torch.Tensor of shape (out_features, in_features).
        shuffle_type: one of the BipartiteRandomization methods ('random', 'random_l_links', etc.)
        num_swaps: for methods requiring swap counts
        is_sparse: whether to use sparse routines

    Returns:
        Shuffled weight matrix.
    """
    # Convert to edgelist
    out_f, in_f = weight_matrix.shape
    edges = [(j, i, float(weight_matrix[i, j].item()))
             for i in range(out_f) for j in range(in_f)]

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
        edges = BipartiteRandomization.randomize_edges_with_sign_separation_fraction_rewire(edges,
                                                                                          num_swaps)
    elif shuffle_type in ('keep_total_strength_left', 'keep_total_strength_right',
                          'keep_pos_neg_strength_left', 'keep_pos_neg_strength_right'):
        fn_map = {
            'keep_total_strength_left': 'reshuffle_all_weights_left',
            'keep_total_strength_right': 'reshuffle_all_weights_right',
            'keep_pos_neg_strength_left': 'reshuffle_separate_signs_left',
            'keep_pos_neg_strength_right': 'reshuffle_separate_signs_right'
        }
        func_name = fn_map[shuffle_type]
        if is_sparse:
            fn = getattr(BipartiteRandomizationSparse, func_name)
        else:
            fn = getattr(BipartiteRandomization, func_name)
        edges = fn(edges)
    elif shuffle_type == 'keep_in_out_degree_swap_edges':
        if is_sparse:
            edges, _ = BipartiteRandomizationSparse.positive_edge_switching_algorithm(edges, num_swaps)
        else:
            edges, _ = BipartiteRandomization.positive_edge_switching_algorithm(edges, num_swaps)
    else:
        raise ValueError(f"Unknown shuffle_type '{shuffle_type}'")

    # Rebuild matrix
    new_w = torch.zeros_like(weight_matrix)
    for j, i, val in edges:
        new_w[i, j] = val
    return new_w


def modify_weights(weight_matrix: torch.Tensor,
                   operation: str,
                   **kwargs) -> torch.Tensor:
    """
    Wrapper to apply a specified modification to a weight matrix.

    Args:
        weight_matrix: torch.Tensor
        operation: one of 'noise', 'flip', 'threshold', 'shuffle'
        kwargs: parameters for the chosen operation

    Returns:
        Modified weight matrix.
    """
    op = operation.lower()
    if op == 'noise':
        return apply_noise_to_weights(weight_matrix,
                                      sigma=kwargs.get('sigma', 0.0),
                                      to_signs=kwargs.get('to_signs', False))
    elif op == 'flip':
        return flip_weights(weight_matrix,
                             q=kwargs.get('q', 0.1),
                             method=kwargs.get('method', 'smallest'),
                             to_signs=kwargs.get('to_signs', False))
    elif op == 'threshold':
        return threshold_weights(weight_matrix,
                                 method=kwargs.get('method', 'smallest'),
                                 fraction_non_zero=kwargs.get('fraction_non_zero', 1.0),
                                 to_signs=kwargs.get('to_signs', False))
    elif op == 'shuffle':
        return shuffle_weights(weight_matrix,
                               shuffle_type=kwargs.get('shuffle_type', 'random'),
                               num_swaps=kwargs.get('num_swaps', None),
                               is_sparse=kwargs.get('is_sparse', False))
    else:
        raise ValueError(f"Unsupported operation '{operation}'")