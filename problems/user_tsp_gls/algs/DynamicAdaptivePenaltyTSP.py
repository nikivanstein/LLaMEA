import numpy as np

class DynamicAdaptivePenaltyTSP:
    def __init__(self, alpha, beta, gamma, delta, epsilon, zeta):
        """
        :param alpha: A parameter controlling the penalty strength based on the usage frequency of edges.
        :param beta: A decay parameter that reduces the penalty strength over iterations.
        :param gamma: A parameter for increasing exploration by adding penalties to frequently used edges in the local optimal tour.
        :param delta: A parameter for dynamically adjusting penalties based on the variance of edge usage.
        :param epsilon: A parameter for boosting the penalty for the most frequently used edges.
        :param zeta: A parameter for introducing randomness to escape local optima.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        """
        Updates the edge distances to avoid being trapped in the local optimum.
        
        :param edge_distance: 2D numpy array representing the edge distances.
        :param local_opt_tour: 1D numpy array representing the current local optimal tour.
        :param edge_n_used: 2D numpy array representing the number of times each edge has been used.
        :return: 2D numpy array representing the updated edge distances.
        """
        n = edge_distance.shape[0]
        updated_edge_distance = np.copy(edge_distance)

        # Compute the variance of edge usage for dynamic adjustment
        edge_usage_variance = np.var(edge_n_used)

        # Determine the most frequently used edges
        max_usage = np.max(edge_n_used)

        # Apply penalties to edges based on their usage frequency and the local optimal tour
        for i in range(len(local_opt_tour) - 1):
            u, v = local_opt_tour[i], local_opt_tour[i + 1]
            usage_penalty = self.alpha * (edge_n_used[u, v] + self.gamma) + self.delta * edge_usage_variance

            # Boost penalty for the most frequently used edges
            if edge_n_used[u, v] == max_usage:
                usage_penalty *= self.epsilon

            updated_edge_distance[u, v] += usage_penalty
            updated_edge_distance[v, u] += usage_penalty  # Since the matrix is symmetric

        # Apply a decay to the previous penalties to allow exploration of other paths
        decayed_penalty = np.exp(-self.beta * edge_n_used)
        updated_edge_distance = updated_edge_distance * decayed_penalty

        # Introduce randomness to help escape local optima
        random_penalty = np.random.rand(n, n) * self.zeta
        updated_edge_distance += random_penalty

        return updated_edge_distance

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # 1.850371707708594e-14
    config = {'alpha': 0.5368935257196, 'beta': 0.3117611007765, 'delta': 1.6591668916866, 'epsilon': 0.9257950854488, 'gamma': 1.6687361944467, 'zeta': 0.4151510921959}
    scoringalg = DynamicAdaptivePenaltyTSP(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)