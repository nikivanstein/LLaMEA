import numpy as np

class OptimizedDynamicPenaltyAdjustment:
    def __init__(self, distance_weight, penalty_factor, usage_weight, penalty_decay, penalty_boost, min_penalty, max_penalty, adaptive_factor, adaptive_threshold, non_penalized_weight):
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

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        N = edge_distance.shape[0]
        updated_edge_distance = edge_distance.copy()

        # Penalty adjustments with dynamic penalty decay and boost
        for i in range(N):
            for j in range(i + 1, N):
                if edge_n_used[i, j] > 0:
                    used_penalty = self.usage_weight * edge_n_used[i, j]
                    distance_penalty = self.distance_weight * edge_distance[i, j]
                    total_penalty = self.penalty_factor * (used_penalty + distance_penalty)
                    total_penalty = np.clip(total_penalty, self.min_penalty, self.max_penalty)
                    updated_edge_distance[i, j] += total_penalty
                    updated_edge_distance[j, i] += total_penalty
                else:
                    updated_edge_distance[i, j] += self.non_penalized_weight
                    updated_edge_distance[j, i] += self.non_penalized_weight

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
    # 1.850371707708594e-14
    config = {'adaptive_factor': 0.1730769457154, 'adaptive_threshold': 1.213661954511, 'distance_weight': 1.9481584954082, 'max_penalty': 1.4907584040949, 'min_penalty': 0.0633670543234, 'non_penalized_weight': 1.8767456610247, 'penalty_boost': 1.6347040478582, 'penalty_decay': 0.8454405608201, 'penalty_factor': 3.9730388285514, 'usage_weight': 1.2151927824108}
    scoringalg = OptimizedDynamicPenaltyAdjustment(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)