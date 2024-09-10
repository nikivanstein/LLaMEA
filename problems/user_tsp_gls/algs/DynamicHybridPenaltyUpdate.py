import numpy as np

class DynamicHybridPenaltyUpdate:
    def __init__(self, distance_weight, penalty_factor, usage_weight, penalty_decay, penalty_boost, min_penalty, max_penalty, adaptive_factor, adaptive_threshold, non_penalized_weight, hybrid_factor, exploration_bias):
        self.distance_weight = distance_weight
        self.penalty_factor = penalty_factor
        self.usage_weight = usage_weight
        self.penalty_decay = penalty_decay
        self.penalty_boost = penalty_boost
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.adaptive_factor = adaptive_factor
        self.adaptive_threshold = adaptive_threshold
        self.non_penalized_weight = non_penalized_weight
        self.hybrid_factor = hybrid_factor
        self.exploration_bias = exploration_bias

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        N = edge_distance.shape[0]
        updated_edge_distance = edge_distance.copy()

        # Hybrid penalty adjustments based on both usage and distance
        for i in range(N):
            for j in range(i + 1, N):
                if edge_n_used[i, j] > 0:
                    used_penalty = self.usage_weight * edge_n_used[i, j]
                    distance_penalty = self.distance_weight * edge_distance[i, j]
                    hybrid_penalty = self.hybrid_factor * (used_penalty + distance_penalty)
                    total_penalty = self.penalty_factor * hybrid_penalty
                    total_penalty = np.clip(total_penalty, self.min_penalty, self.max_penalty)
                    updated_edge_distance[i, j] += total_penalty
                    updated_edge_distance[j, i] += total_penalty
                else:
                    # Introduce exploration bias for non-used edges
                    updated_edge_distance[i, j] += self.non_penalized_weight + self.exploration_bias
                    updated_edge_distance[j, i] += self.non_penalized_weight + self.exploration_bias

        # Adaptive penalties based on local optimal tour
        for idx in range(len(local_opt_tour) - 1):
            i, j = local_opt_tour[idx], local_opt_tour[idx + 1]
            adaptive_penalty = self.penalty_factor * (1 + self.adaptive_factor * edge_n_used[i, j])
            adaptive_penalty = np.clip(adaptive_penalty, self.min_penalty, self.max_penalty)
            updated_edge_distance[i, j] += adaptive_penalty * edge_distance[i, j]
            updated_edge_distance[j, i] += adaptive_penalty * edge_distance[i, j]

        # Dynamic adjustment based on penalty values and edge usage
        for i in range(N):
            for j in range(i + 1, N):
                current_penalty = updated_edge_distance[i, j] - edge_distance[i, j]
                if current_penalty > self.adaptive_threshold:
                    updated_edge_distance[i, j] *= self.penalty_decay
                    updated_edge_distance[j, i] *= self.penalty_decay
                else:
                    updated_edge_distance[i, j] *= self.penalty_boost
                    updated_edge_distance[j, i] *= self.penalty_boost

        # Ensure non-negative distances
        updated_edge_distance = np.maximum(updated_edge_distance, 0)

        return updated_edge_distance

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # 1.1102230246251565e-14
    config = {'adaptive_factor': 3.9985866627964, 'adaptive_threshold': 0.2051534938112, 'distance_weight': 1.4025019387862, 'exploration_bias': 0.2290180521214, 'hybrid_factor': 1.2022412709054, 'max_penalty': 1.5141351866239, 'min_penalty': 0.3797514452922, 'non_penalized_weight': 2.9930269691928, 'penalty_boost': 1.7671010868896, 'penalty_decay': 0.5007060335101, 'penalty_factor': 3.3801992309288, 'usage_weight': 3.3407277004498}
    scoringalg = DynamicHybridPenaltyUpdate(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)