import numpy as np

class EnhancedMemoryAdaptivePenaltyTSP:
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1, max_penalty=10.0, min_penalty=0.1, decay_rate=0.1, elite_frac=0.1, feedback_weight=0.5, penalty_scaling=0.01, memory_weight=0.9, exploration_weight=0.05, memory_decay=0.95):
        """
        :param alpha: Parameter adjusting the penalty strength based on edge usage.
        :param beta: Decay parameter reducing the penalty strength over iterations.
        :param gamma: Factor for increasing exploration by adding penalties to frequently used edges.
        :param max_penalty: Maximum allowed penalty to avoid excessive increase in edge distances.
        :param min_penalty: Minimum allowed penalty to avoid zero penalties.
        :param decay_rate: Rate at which the penalties decay over iterations.
        :param elite_frac: Fraction of top edges to preserve without penalty.
        :param feedback_weight: Weight of the feedback mechanism based on overall tour quality.
        :param penalty_scaling: Scaling factor for penalties based on historical memory.
        :param memory_weight: Weight applied to the influence of the memory on updated distances.
        :param exploration_weight: Weight applied to enhance exploration based on edge usage.
        :param memory_decay: Decay rate for the historical memory to adapt over iterations.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_penalty = max_penalty
        self.min_penalty = min_penalty
        self.decay_rate = decay_rate
        self.elite_frac = elite_frac
        self.feedback_weight = feedback_weight
        self.penalty_scaling = penalty_scaling
        self.memory_weight = memory_weight
        self.exploration_weight = exploration_weight
        self.memory_decay = memory_decay
        self.memory = None

    def initialize_memory(self, n):
        self.memory = np.zeros((n, n))

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        """
        Updates the edge distances to avoid being trapped in the local optimum.
        
        :param edge_distance: 2D numpy array representing the edge distances.
        :param local_opt_tour: 1D numpy array representing the current local optimal tour.
        :param edge_n_used: 2D numpy array representing the number of times each edge has been used.
        :return: 2D numpy array representing the updated edge distances.
        """
        n = edge_distance.shape[0]
        if self.memory is None:
            self.initialize_memory(n)
        
        updated_edge_distance = np.copy(edge_distance)
        penalty_matrix = np.zeros_like(edge_distance)

        # Apply penalties to edges based on their usage frequency and the local optimal tour
        for i in range(len(local_opt_tour) - 1):
            u, v = local_opt_tour[i], local_opt_tour[i + 1]
            usage_penalty = self.alpha * np.log(1 + edge_n_used[u, v])  # Logarithmic penalty to smoothen the impact

            # Cap the penalty to avoid excessive increase
            usage_penalty = min(max(usage_penalty, self.min_penalty), self.max_penalty)

            penalty_matrix[u, v] += usage_penalty
            penalty_matrix[v, u] += usage_penalty  # Since the matrix is symmetric

        # Apply a non-linear decay to the previous penalties
        penalty_matrix *= np.exp(-self.beta * np.sqrt(edge_n_used))

        # Dynamic exploration factor to further incentivize exploration
        exploration_factor = self.gamma * (np.max(edge_n_used) - edge_n_used)
        penalty_matrix += self.exploration_weight * exploration_factor

        # Normalize the updated edge distances to maintain consistency
        min_distance = np.min(penalty_matrix[np.nonzero(penalty_matrix)])
        max_distance = np.max(penalty_matrix)
        normalized_penalty_matrix = (penalty_matrix - min_distance) / (max_distance - min_distance + 1e-6)

        updated_edge_distance += normalized_penalty_matrix

        # Apply overall decay to penalties to avoid runaway increases over long iterations
        updated_edge_distance *= (1 - self.decay_rate)

        # Preserve a fraction of edges as elite to avoid excessive penalties
        flat_distances = updated_edge_distance.flatten()
        elite_threshold = np.percentile(flat_distances, self.elite_frac * 100)
        elite_mask = updated_edge_distance < elite_threshold
        updated_edge_distance[elite_mask] = edge_distance[elite_mask]

        # Feedback mechanism based on the overall quality of the tour
        tour_length = sum(edge_distance[local_opt_tour[i], local_opt_tour[i + 1]] for i in range(len(local_opt_tour) - 1))
        feedback_penalty = self.feedback_weight * (tour_length - np.mean(edge_distance))
        if feedback_penalty > 0:
            penalty_matrix += feedback_penalty

        # Update memory with decay
        if self.memory is not None:
            self.memory = self.memory * self.memory_decay + penalty_matrix

        # Apply historical memory to the updated edge distances
        if self.memory is not None:
            updated_edge_distance += self.penalty_scaling * self.memory

        return updated_edge_distance

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # 1.850371707708594e-14
    config = {'alpha': 1.7521767024882, 'beta': 0.5212888342328, 'decay_rate': 0.3570327109471, 'elite_frac': 0.3696445946861, 'exploration_weight': 0.2110755836405, 'feedback_weight': 0.7284060320817, 'gamma': 0.0680560386274, 'max_penalty': 6.6297375643626, 'memory_decay': 0.8758478531614, 'memory_weight': 0.1854300661944, 'min_penalty': 0.6977903513983, 'penalty_scaling': 0.0474904146092}
    scoringalg = EnhancedMemoryAdaptivePenaltyTSP(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)