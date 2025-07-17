import random


class BipartiteRandomizationSparse:

    @staticmethod
    def get_randomization(type):
        match type:
            case 'random': return BipartiteRandomizationSparse.shuffle_weights
            case 'random_pos_neg': return BipartiteRandomizationSparse.randomize_edges_with_sign_separation
            case 'keep_total_strength_left': return BipartiteRandomizationSparse.reshuffle_all_weights_left
            case 'keep_total_strength_right': return BipartiteRandomizationSparse.reshuffle_all_weights_right
            case 'keep_pos_neg_strength_left': return BipartiteRandomizationSparse.reshuffle_separate_signs_left
            case 'keep_pos_neg_strength_right': return BipartiteRandomizationSparse.reshuffle_separate_signs_right
            case 'keep_in_out_degree_swap_edges': return BipartiteRandomizationSparse.positive_edge_switching_algorithm


    @staticmethod
    def shuffle_weights(edges):
        zero_weight_edges = [(u, v, w) for u, v, w in edges if w == 0]
        non_zero_edges = [(u, v, w) for u, v, w in edges if w != 0]
        non_zero_weights = [w for _, _, w in non_zero_edges]
        random.shuffle(non_zero_weights)
        shuffled_non_zero_edges = [(u, v, w) for (u, v, _), w in zip(non_zero_edges, non_zero_weights)]
        return shuffled_non_zero_edges + zero_weight_edges


    @staticmethod
    def reshuffle_separate_signs_left(edges):
        """
        Shuffle positive and negative weights separately for each LEFT node,
        preserving zero-weight edges and keeping the total strength intact.

        Parameters:
        edges (list of tuples): List of edges as (u, v, weight)

        Returns:
        list of tuples: Edges with separately shuffled positive and negative weights
        """
        # Group edges by LEFT node and categorize weights
        node_weights = {}
        zero_weight_edges = []

        for u, v, w in edges:
            if w == 0:
                zero_weight_edges.append((u, v, w))
            else:
                if u not in node_weights:
                    node_weights[u] = {"positive": [], "negative": []}
                if w > 0:
                    node_weights[u]["positive"].append((v, w))
                else:
                    node_weights[u]["negative"].append((v, w))

        # Shuffle positive and negative weights separately for each node
        shuffled_edges = []
        for u, weights_dict in node_weights.items():
            positive_weights = [w for _, w in weights_dict["positive"]]
            negative_weights = [w for _, w in weights_dict["negative"]]

            random.shuffle(positive_weights)
            random.shuffle(negative_weights)

            # Reassign shuffled weights to edges
            for i, (v, _) in enumerate(weights_dict["positive"]):
                shuffled_edges.append((u, v, positive_weights[i]))
            for i, (v, _) in enumerate(weights_dict["negative"]):
                shuffled_edges.append((u, v, negative_weights[i]))

        # Combine shuffled edges with zero-weight edges
        return shuffled_edges + zero_weight_edges


    @staticmethod
    def reshuffle_separate_signs_right(edges):
        """
        Shuffle positive and negative weights separately for each RIGHT node,
        preserving zero-weight edges and keeping the total strength intact.

        Parameters:
        edges (list of tuples): List of edges as (u, v, weight)

        Returns:
        list of tuples: Edges with separately shuffled positive and negative weights
        """
        # Group edges by RIGHT node and categorize weights
        node_weights = {}
        zero_weight_edges = []

        for u, v, w in edges:
            if w == 0:
                zero_weight_edges.append((u, v, w))
            else:
                if v not in node_weights:
                    node_weights[v] = {"positive": [], "negative": []}
                if w > 0:
                    node_weights[v]["positive"].append((u, w))
                else:
                    node_weights[v]["negative"].append((u, w))

        # Shuffle positive and negative weights separately for each RIGHT node
        shuffled_edges = []
        for v, weights_dict in node_weights.items():
            positive_weights = [w for _, w in weights_dict["positive"]]
            negative_weights = [w for _, w in weights_dict["negative"]]

            random.shuffle(positive_weights)
            random.shuffle(negative_weights)

            # Reassign shuffled weights to edges
            for i, (u, _) in enumerate(weights_dict["positive"]):
                shuffled_edges.append((u, v, positive_weights[i]))
            for i, (u, _) in enumerate(weights_dict["negative"]):
                shuffled_edges.append((u, v, negative_weights[i]))

        # Combine shuffled edges with zero-weight edges
        return shuffled_edges + zero_weight_edges


    @staticmethod
    def reshuffle_all_weights_left(edges):
        """
        Shuffle all nonzero weights together for each input LEFT node,
        keeping the total weight (sum of all edges) for that node intact
        but leaving any weight==0 in its original place.
        """
        # Group edges by node and store all (node2, weight) pairs together
        node_weights = {}
        for node1, node2, weight in edges:
            if node1 not in node_weights:
                node_weights[node1] = []
            node_weights[node1].append((node2, weight))

        shuffled_edges = []
        # Shuffle all NON-zero weights for each node
        for node1, weights_list in node_weights.items():
            # Extract only the nonzero weights
            nonzero_weights = [w for (_, w) in weights_list if w != 0]

            # Shuffle the nonzero weights
            random.shuffle(nonzero_weights)

            # Reassign the shuffled nonzero weights, skipping zero slots
            idx_nonzero = 0
            node_shuffled = []
            for (node2, w) in weights_list:
                if w == 0:
                    # Keep zero in the same place
                    node_shuffled.append((node1, node2, 0))
                else:
                    # Take the next shuffled nonzero weight
                    new_w = nonzero_weights[idx_nonzero]
                    idx_nonzero += 1
                    node_shuffled.append((node1, node2, new_w))

            # Append to the global list of edges
            shuffled_edges.extend(node_shuffled)

        return shuffled_edges

    @staticmethod
    def reshuffle_all_weights_right(edges):
        """
        Shuffle all nonzero weights together for each "RIGHT" node,
        while leaving any weight == 0 in its original place.
        """
        # Group edges by right node and store (left_node, weight) pairs
        node_weights = {}
        for node1, node2, weight in edges:
            if node2 not in node_weights:
                node_weights[node2] = []
            node_weights[node2].append((node1, weight))

        shuffled_edges = []
        # For each right node, shuffle only the nonzero weights
        for node2, weights_list in node_weights.items():
            # Extract nonzero weights
            nonzero_weights = [w for (_, w) in weights_list if w != 0]

            # Shuffle the nonzero weights in place
            random.shuffle(nonzero_weights)

            # Reassign weights, skipping over zero positions
            idx_nonzero = 0
            node_shuffled = []
            for (node1, w) in weights_list:
                if w == 0:
                    # Keep zero in the same place
                    node_shuffled.append((node1, node2, 0))
                else:
                    # Take the next shuffled nonzero weight
                    new_w = nonzero_weights[idx_nonzero]
                    idx_nonzero += 1
                    node_shuffled.append((node1, node2, new_w))

            # Collect shuffled edges for this right node
            shuffled_edges.extend(node_shuffled)

        return shuffled_edges

    @staticmethod
    def randomize_edges_with_sign_separation(edges):
        """
        Randomize the weights of edges by shuffling positive and negative weights separately,
        while preserving zero-weight edges and the structure of the bipartite graph.

        Parameters:
        edges (list of tuples): List of edges as (u, v, weight)

        Returns:
        list of tuples: Edges with separately shuffled positive and negative weights
        """
        # Separate edges based on weight sign and zero weights
        positive_edges = [(u, v, w) for u, v, w in edges if w > 0]
        negative_edges = [(u, v, w) for u, v, w in edges if w < 0]
        zero_weight_edges = [(u, v, w) for u, v, w in edges if w == 0]

        # Extract weights for shuffling
        positive_weights = [w for _, _, w in positive_edges]
        negative_weights = [w for _, _, w in negative_edges]

        # Shuffle positive and negative weights
        random.shuffle(positive_weights)
        random.shuffle(negative_weights)

        # Reassign shuffled weights to edges
        shuffled_positive_edges = [(u, v, w) for (u, v, _), w in zip(positive_edges, positive_weights)]
        shuffled_negative_edges = [(u, v, w) for (u, v, _), w in zip(negative_edges, negative_weights)]

        # Combine all edges: shuffled positive, shuffled negative, and preserved zero-weight edges
        return shuffled_positive_edges + shuffled_negative_edges + zero_weight_edges


    @staticmethod
    def positive_edge_switching_algorithm(edges, num_swaps=None):
        """
        Randomize the graph using the edge-switching algorithm while preserving
        the in-degree and out-degree distributions.

        Parameters:
        edges (list of tuples): List of edges as (u, v, weight)

        Returns:
        list of tuples: Randomized list of edges
        """

        positive_edges = [(u, v, w) for u, v, w in edges if w > 0]
        edge_set = set((u, v) for u, v, _ in positive_edges)
        randomized_edges = positive_edges[:]

        zero_weight_edges = [(u, v, w) for u, v, w in edges if w == 0]
        zero_weight_set = set((u, v) for u, v, _ in zero_weight_edges)


        if len(randomized_edges) < 2: # cannot rewire
            return edges, 0

        if num_swaps is None:
            num_swaps = 10 * len(randomized_edges)

        rewired_links = 0

        for _ in range(num_swaps):
            (u1, v1, w1), (u2, v2, w2) = random.sample(randomized_edges, 2)

            new_edge1 = (u1, v2)
            new_edge2 = (u2, v1)

            # Check validity of the swap (no duplicate edges or self-loops or rewire to zero-weight edge)
            if (
                new_edge1 not in zero_weight_set
                and new_edge2 not in zero_weight_set
                and new_edge1 not in edge_set
                and new_edge2 not in edge_set
                and u1 != v2
                and u2 != v1
            ):
                # Perform the swap
                edge_set.remove((u1, v1))
                edge_set.remove((u2, v2))
                edge_set.add(new_edge1)
                edge_set.add(new_edge2)

                # Update the edge list with new swapped edges
                randomized_edges.remove((u1, v1, w1))
                randomized_edges.remove((u2, v2, w2))
                randomized_edges.append((u1, v2, w1))
                randomized_edges.append((u2, v1, w2))

                rewired_links += 1
                if rewired_links >= len(positive_edges):
                    break # already rewired all links

        # Fill negative connections now
        negative_weights = [w for _, _, w in edges if w < 0]
        # random.shuffle(negative_weights)

        negative_edge_dict = {(u, v): w for u, v, w in edges if w <= 0}
        randomized_edge_dict = {(u, v): w for u, v, w in randomized_edges}

        full_edge_set = set((u, v) for u, v, _ in edges)
        all_edges = []
        idx = 0
        for u, v in full_edge_set:
            if (u, v) not in randomized_edge_dict:
                if (u, v) in negative_edge_dict:
                    all_edges.append((u, v, negative_edge_dict[(u, v)]))
                else:    
                    all_edges.append((u, v, negative_weights[idx]))
                    negative_edge_dict.pop((u, v), None)
                    idx += 1
            else:
                all_edges.append((u, v, randomized_edge_dict[(u, v)]))

        return all_edges, (rewired_links / len(positive_edges))
