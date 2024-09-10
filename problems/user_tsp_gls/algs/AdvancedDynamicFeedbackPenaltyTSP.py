import numpy as np

class AdvancedDynamicFeedbackPenaltyTSP:
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1, delta=0.01, feedback_weight=0.5, reward_weight=0.5, max_penalty=10.0, min_penalty=0.1):
        """
        :param alpha: Parameter adjusting the penalty strength based on edge usage.
        :param beta: Decay parameter reducing the penalty strength over iterations.
        :param gamma: Factor for increasing exploration by adding penalties to frequently used edges.
        :param delta: Small constant to avoid division by zero.
        :param feedback_weight: Weight of the feedback mechanism based on overall tour quality.
        :param reward_weight: Weight of the reward mechanism to reinforce good edges.
        :param max_penalty: Maximum allowed penalty to avoid excessive increase in edge distances.
        :param min_penalty: Minimum allowed penalty to avoid zero penalties.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.feedback_weight = feedback_weight
        self.reward_weight = reward_weight
        self.max_penalty = max_penalty
        self.min_penalty = min_penalty

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        """
        Updates the edge distances to avoid being trapped in the local optimum.
        
        :param edge_distance: 2D numpy array representing the edge distances.
        :param local_opt_tour: 1D numpy array representing the current local optimal tour.
        :param edge_n_used: 2D numpy array representing the number of times each edge has been used.
        :return: 2D numpy array representing the updated edge distances.
        """
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
        penalty_matrix += exploration_factor

        # Normalize the updated edge distances to maintain consistency
        min_distance = np.min(penalty_matrix[np.nonzero(penalty_matrix)])
        max_distance = np.max(penalty_matrix)
        normalized_penalty_matrix = (penalty_matrix - min_distance) / (max_distance - min_distance + self.delta)

        updated_edge_distance += normalized_penalty_matrix

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
    config = {'alpha': 1.0018309128471, 'beta': 0.9283601141535, 'delta': 0.0888253703332, 'feedback_weight': 0.3470206518658, 'gamma': 0.0830076882429, 'max_penalty': 4.5355474697426, 'min_penalty': 0.6188640497997, 'reward_weight': 0.4011157567613}
    scoringalg = AdvancedDynamicFeedbackPenaltyTSP(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)