import numpy as np

class EnhancedStochasticEdgeReweightingTSP:
    def __init__(self, alpha, beta, gamma, delta, epsilon, zeta, eta, theta):
        """
        :param alpha: Parameter controlling the penalty strength based on the usage frequency of edges.
        :param beta: Decay parameter that reduces the penalty strength over iterations.
        :param gamma: Parameter for increasing exploration by adding penalties to frequently used edges in the local optimal tour.
        :param delta: Parameter for dynamically adjusting penalties based on the variance of edge usage.
        :param epsilon: Parameter for boosting the penalty for the most frequently used edges.
        :param zeta: Parameter for introducing randomness to escape local optima.
        :param eta: Parameter for introducing global penalties to balance exploration and exploitation.
        :param theta: Parameter for adjusting penalties based on the distance of edges in the local optimum.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
        self.eta = eta
        self.theta = theta

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

            # Adjust penalty based on the distance of the edge in the local optimal tour
            distance_penalty = self.theta * edge_distance[u, v]
            usage_penalty += distance_penalty

            updated_edge_distance[u, v] += usage_penalty
            updated_edge_distance[v, u] += usage_penalty  # Since the matrix is symmetric

        # Apply a decay to the previous penalties
        decay_factor = np.exp(-self.beta * edge_n_used)
        updated_edge_distance = updated_edge_distance * decay_factor

        # Introduce randomness to help escape local optima
        random_penalty = np.random.rand(n, n) * self.zeta
        random_penalty = (random_penalty + random_penalty.T) / 2  # Ensure symmetry
        updated_edge_distance += random_penalty

        # Introduce a global penalty to balance exploration and exploitation
        global_penalty = np.sum(edge_n_used) / (n * (n - 1)) * self.eta
        updated_edge_distance += global_penalty

        return updated_edge_distance

# Space:
configuration_space = {
    "alpha": (0.1, 2.0),
    "beta": (0.1, 1.0),
    "gamma": (0.0, 1.0),
    "delta": (0.0, 1.0),
    "epsilon": (1.0, 10.0),
    "zeta": (0.0, 0.5),
    "eta": (0.0, 1.0),
    "theta": (0.0, 2.0)
}

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # 1.850371707708594e-14
    config = {'alpha': 0.3953434264287, 'beta': 0.4501514552161, 'delta': 0.7934444304556, 'epsilon': 8.9579637339339, 'eta': 0.7152201170102, 'gamma': 0.2464154073969, 'theta': 0.1363251972944, 'zeta': 0.3225747565739}
    scoringalg = EnhancedStochasticEdgeReweightingTSP(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)