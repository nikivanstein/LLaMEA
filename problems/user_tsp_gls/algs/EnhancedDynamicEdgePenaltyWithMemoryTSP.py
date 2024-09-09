import numpy as np

class EnhancedDynamicEdgePenaltyWithMemoryTSP:
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1, delta=0.01, exploration_weight=0.2, feedback_weight=0.5, reward_weight=0.5, max_penalty=10.0, min_penalty=0.1, epsilon=0.05, memory_decay=0.9):
        """
        :param alpha: Parameter adjusting the penalty strength based on edge usage.
        :param beta: Decay parameter reducing the penalty strength over iterations.
        :param gamma: Factor for increasing exploration by adding penalties to frequently used edges.
        :param delta: Small constant to avoid division by zero.
        :param exploration_weight: Weight for the exploratory penalty to encourage diverse routes.
        :param feedback_weight: Weight of the feedback mechanism based on overall tour quality.
        :param reward_weight: Weight of the reward mechanism to reinforce good edges.
        :param max_penalty: Maximum allowed penalty to avoid excessive increase in edge distances.
        :param min_penalty: Minimum allowed penalty to avoid zero penalties.
        :param epsilon: Small value added to ensure diversity in penalties.
        :param memory_decay: Decay rate for memory of past penalties.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.exploration_weight = exploration_weight
        self.feedback_weight = feedback_weight
        self.reward_weight = reward_weight
        self.max_penalty = max_penalty
        self.min_penalty = min_penalty
        self.epsilon = epsilon
        self.memory_decay = memory_decay
        self.penalty_memory = None

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        """
        Updates the edge distances to avoid being trapped in the local optimum.
        
        :param edge_distance: 2D numpy array representing the edge distances.
        :param local_opt_tour: 1D numpy array representing the current local optimal tour.
        :param edge_n_used: 2D numpy array representing the number of times each edge has been used.
        :return: 2D numpy array representing the updated edge distances.
        """
        N = edge_distance.shape[0]
        updated_edge_distance = np.copy(edge_distance)
        penalty_matrix = np.zeros_like(edge_distance)

        # Initialize penalty memory if it doesn't exist
        if self.penalty_memory is None:
            self.penalty_memory = np.zeros((N, N))

        # Apply penalties to edges based on their usage frequency and the local optimal tour
        for i in range(len(local_opt_tour) - 1):
            u, v = local_opt_tour[i], local_opt_tour[i + 1]
            usage_penalty = self.alpha * np.log(1 + edge_n_used[u, v]) + self.epsilon  # Logarithmic penalty to smoothen the impact
            
            # Cap the penalty to avoid excessive increase
            usage_penalty = min(max(usage_penalty, self.min_penalty), self.max_penalty)
            
            penalty_matrix[u, v] += usage_penalty
            penalty_matrix[v, u] += usage_penalty  # Since the matrix is symmetric

        # Apply a non-linear decay to the previous penalties
        penalty_matrix *= np.exp(-self.beta * np.sqrt(edge_n_used))

        # Dynamic exploration factor to further incentivize exploration
        exploration_factor = self.gamma * (np.max(edge_n_used) - edge_n_used) + self.exploration_weight
        penalty_matrix += exploration_factor

        # Normalize the updated edge distances to maintain consistency
        min_distance = np.min(penalty_matrix[np.nonzero(penalty_matrix)])
        max_distance = np.max(penalty_matrix)
        normalized_penalty_matrix = (penalty_matrix - min_distance) / (max_distance - min_distance + self.delta)

        # Update penalty memory with decay
        self.penalty_memory = self.memory_decay * self.penalty_memory + (1 - self.memory_decay) * normalized_penalty_matrix

        updated_edge_distance += self.penalty_memory

        # Feedback mechanism based on the overall quality of the tour
        tour_length = sum(edge_distance[local_opt_tour[i], local_opt_tour[i + 1]] for i in range(len(local_opt_tour) - 1))
        feedback_penalty = self.feedback_weight * (tour_length - np.mean(edge_distance))
        if feedback_penalty > 0:
            penalty_matrix += feedback_penalty

        # Reward mechanism to reinforce good edges
        reward_matrix = np.zeros_like(edge_distance)
        for i in range(len(local_opt_tour) - 1):
            u, v = local_opt_tour[i], local_opt_tour[i + 1]
            reward_matrix[u, v] += self.reward_weight
            reward_matrix[v, u] += self.reward_weight  # Since the matrix is symmetric

        penalty_matrix -= reward_matrix

        updated_edge_distance += penalty_matrix

        return updated_edge_distance

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # 1.850371707708594e-14
    config = {'alpha': 1.819667665381, 'beta': 0.4385085600428, 'delta': 0.0474453113871, 'epsilon': 0.0955881857686, 'exploration_weight': 0.7869258807972, 'feedback_weight': 0.785988750495, 'gamma': 0.1124158192519, 'max_penalty': 12.3598325280473, 'memory_decay': 0.7611037363485, 'min_penalty': 0.1903303262778, 'reward_weight': 0.3521109447815}
    scoringalg = EnhancedDynamicEdgePenaltyWithMemoryTSP(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)