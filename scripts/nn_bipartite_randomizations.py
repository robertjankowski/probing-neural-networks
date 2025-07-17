import random


class BipartiteRandomization:

    @staticmethod
    def get_randomization(type):
        match type:
            case 'random': return BipartiteRandomization.shuffle_weights
            case 'random_l_links': return BipartiteRandomization.shuffle_l_weights
            case 'random_pos_neg': return BipartiteRandomization.randomize_edges_with_sign_separation
            case 'random_pos_neg_fraction': return BipartiteRandomization.randomize_edges_with_sign_separation_fraction_rewire
            case 'keep_total_strength_left': return BipartiteRandomization.reshuffle_all_weights_left
            case 'keep_total_strength_right': return BipartiteRandomization.reshuffle_all_weights_right
            case 'keep_pos_neg_strength_left': return BipartiteRandomization.reshuffle_separate_signs_left
            case 'keep_pos_neg_strength_right': return BipartiteRandomization.reshuffle_separate_signs_right
            case 'keep_in_out_degree_swap_edges': return BipartiteRandomization.positive_edge_switching_algorithm


    @staticmethod
    def shuffle_weights(edges):
        weights = [w for (_, _, w) in edges]
        random.shuffle(weights)
        edges = [(u, v, w) for ((u, v, _), w) in zip(edges, weights)]
        return edges

    @staticmethod
    def shuffle_l_weights(edges, L):
        # Randomly sample L edges to shuffle
        sampled_indices = random.sample(range(len(edges)), L)
        sampled_edges = [edges[i] for i in sampled_indices]
        
        weights = [w for (_, _, w) in sampled_edges]
        random.shuffle(weights)
        
        shuffled_sampled_edges = [(u, v, w) for ((u, v, _), w) in zip(sampled_edges, weights)]
        
        edges = [
            shuffled_sampled_edges[sampled_indices.index(i)] if i in sampled_indices else edges[i]
            for i in range(len(edges))
        ]
    
        return edges

    @staticmethod
    def reshuffle_separate_signs_left(edges):
        """
        Shuffle positive and negative weights separately for each input LEFT node,
        keeping the strength (total weight) intact.
        """
        # Group edges by node and separate positive and negative weights
        node_weights = {}
        for node1, node2, weight in edges:
            if node1 not in node_weights:
                node_weights[node1] = {"positive": [], "negative": []}
            if weight > 0:
                node_weights[node1]["positive"].append((node2, weight))
            else:
                node_weights[node1]["negative"].append((node2, weight))

        # Shuffle weights within positive and negative lists for each node
        shuffled_edges = []
        for node1, weights_dict in node_weights.items():
            positive_weights = [weight for _, weight in weights_dict["positive"]]
            negative_weights = [weight for _, weight in weights_dict["negative"]]

            random.shuffle(positive_weights)
            random.shuffle(negative_weights)

            # Reassign the shuffled weights back to the edges
            for i, (node2, _) in enumerate(weights_dict["positive"]):
                shuffled_edges.append((node1, node2, positive_weights[i]))
            for i, (node2, _) in enumerate(weights_dict["negative"]):
                shuffled_edges.append((node1, node2, negative_weights[i]))

        return shuffled_edges

    @staticmethod
    def reshuffle_separate_signs_right(edges):
        """
        Shuffle positive and negative weights separately for each input RIGHT node,
        keeping the strength (total weight) intact.
        """
        # Group edges by node and separate positive and negative weights
        node_weights = {}
        for node1, node2, weight in edges:
            if node2 not in node_weights:
                node_weights[node2] = {"positive": [], "negative": []}
            if weight > 0:
                node_weights[node2]["positive"].append((node1, weight))
            else:
                node_weights[node2]["negative"].append((node1, weight))

        shuffled_edges = []
        for node2, weights_dict in node_weights.items():
            positive_weights = [weight for _, weight in weights_dict["positive"]]
            negative_weights = [weight for _, weight in weights_dict["negative"]]

            random.shuffle(positive_weights)
            random.shuffle(negative_weights)

            # Reassign the shuffled weights back to the edges
            for i, (node1, _) in enumerate(weights_dict["positive"]):
                shuffled_edges.append((node1, node2, positive_weights[i]))
            for i, (node1, _) in enumerate(weights_dict["negative"]):
                shuffled_edges.append((node1, node2, negative_weights[i]))

        return shuffled_edges

    @staticmethod
    def reshuffle_all_weights_left(edges):
        """
        Shuffle all weights (both positive and negative) together for each input LEFT node,
        keeping the strength (total weight) intact.

        """
        # Group edges by node and store all weights together
        node_weights = {}
        for node1, node2, weight in edges:
            if node1 not in node_weights:
                node_weights[node1] = []
            node_weights[node1].append((node2, weight))

        # Shuffle all weights for each node
        shuffled_edges = []
        for node1, weights_list in node_weights.items():
            weights = [weight for _, weight in weights_list]

            random.shuffle(weights)

            # Reassign the shuffled weights back to the edges
            for i, (node2, _) in enumerate(weights_list):
                shuffled_edges.append((node1, node2, weights[i]))

        return shuffled_edges

    @staticmethod
    def reshuffle_all_weights_right(edges):
        """
        Shuffle all weights (both positive and negative) together for each input RIGHT node,
        keeping the strength (total weight) intact.

        """
        # Group edges by node and store all weights together
        node_weights = {}
        for node1, node2, weight in edges:
            if node2 not in node_weights:
                node_weights[node2] = []
            node_weights[node2].append((node1, weight))

        # Shuffle all weights for each node
        shuffled_edges = []
        for node2, weights_list in node_weights.items():
            weights = [weight for _, weight in weights_list]

            random.shuffle(weights)

            # Reassign the shuffled weights back to the edges
            for i, (node1, _) in enumerate(weights_list):
                shuffled_edges.append((node1, node2, weights[i]))

        return shuffled_edges

    @staticmethod
    def randomize_edges_with_sign_separation(edges):
        """
        Randomize the weights of edges by shuffling positive and negative weights separately,
        while keeping the structure of the bipartite graph (node pairs) intact.
        """
        positive_weights = [weight for _, _, weight in edges if weight >= 0]
        negative_weights = [weight for _, _, weight in edges if weight < 0]

        random.shuffle(positive_weights)
        random.shuffle(negative_weights)

        randomized_edges = []
        pos_idx, neg_idx = 0, 0

        for u, v, weight in edges:
            if weight >= 0:
                # Assign a shuffled positive weight
                randomized_edges.append((u, v, positive_weights[pos_idx]))
                pos_idx += 1
            else:
                # Assign a shuffled negative weight
                randomized_edges.append((u, v, negative_weights[neg_idx]))
                neg_idx += 1

        return randomized_edges


    @staticmethod
    def randomize_edges_with_sign_separation_fraction_rewire(edges, num_swaps):
        """
        Randomize the weights of edges by shuffling positive and negative weights separately,
        while keeping the structure of the bipartite graph (node pairs) intact.
        """
        positive_weights = [weight for _, _, weight in edges if weight >= 0]
        negative_weights = [weight for _, _, weight in edges if weight < 0]

        pos_num_to_rewire = int(num_swaps / 2)
        neg_num_to_rewire = num_swaps - pos_num_to_rewire

        if pos_num_to_rewire > len(positive_weights):
            pos_num_to_rewire = len(positive_weights)
        if neg_num_to_rewire > len(negative_weights):
            neg_num_to_rewire = len(negative_weights)

        pos_indices_to_rewire = random.sample(list(range(len(positive_weights))), pos_num_to_rewire)        
        pos_weights_to_rewire = [positive_weights[i] for i in pos_indices_to_rewire]
        random.shuffle(pos_weights_to_rewire)
        for idx, new_weight in zip(pos_indices_to_rewire, pos_weights_to_rewire):
            positive_weights[idx] = new_weight

        neg_indices_to_rewire = random.sample(list(range(len(negative_weights))), neg_num_to_rewire)
        neg_weights_to_rewire = [negative_weights[i] for i in neg_indices_to_rewire]
        random.shuffle(neg_weights_to_rewire)
        for idx, new_weight in zip(neg_indices_to_rewire, neg_weights_to_rewire):
            negative_weights[idx] = new_weight
        
        randomized_edges = []
        pos_idx, neg_idx = 0, 0
        for u, v, weight in edges:
            if weight >= 0:
                # Assign a shuffled positive weight
                randomized_edges.append((u, v, positive_weights[pos_idx]))
                pos_idx += 1
            else:
                # Assign a shuffled negative weight
                randomized_edges.append((u, v, negative_weights[neg_idx]))
                neg_idx += 1

        return randomized_edges


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

        positive_edges = [(u, v, w) for u, v, w in edges if w >= 0]
        edge_set = set((u, v) for u, v, _ in positive_edges)
        randomized_edges = positive_edges[:]

        if num_swaps is None:
            num_swaps = 10 * len(randomized_edges)

        rewired_links = 0

        for _ in range(num_swaps):
            (u1, v1, w1), (u2, v2, w2) = random.sample(randomized_edges, 2)

            new_edge1 = (u1, v2)
            new_edge2 = (u2, v1)

            # Check validity of the swap (no duplicate edges or self-loops)
            if (
                new_edge1 not in edge_set
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

        negative_edge_dict = {(u, v): w for u, v, w in edges if w < 0}
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
