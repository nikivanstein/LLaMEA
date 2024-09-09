import numpy as np

class OptimizedSynergisticDynamicPenaltyUpdate:
    def __init__(self, distance_weight, penalty_factor, usage_weight, penalty_decay, penalty_boost, min_penalty, max_penalty, adaptive_factor, adaptive_threshold, non_penalized_weight, penalty_scale, edge_usage_importance):
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
        self.penalty_scale = penalty_scale
        self.edge_usage_importance = edge_usage_importance

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

        # Adaptive penalties based on local optimal tour with edge usage importance
        for idx in range(len(local_opt_tour) - 1):
            i, j = local_opt_tour[idx], local_opt_tour[idx + 1]
            adaptive_penalty = self.penalty_factor * (1 + self.adaptive_factor * (edge_n_used[i, j] ** self.edge_usage_importance))
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

        # Apply additional scaling to the penalties
        updated_edge_distance *= self.penalty_scale

        # Ensure non-negative distances
        updated_edge_distance = np.maximum(updated_edge_distance, 0)

        return updated_edge_distance

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # -1.1102230246251565e-14
    config = {'adaptive_factor': 3.1661145934231, 'adaptive_threshold': 0.9869204498795, 'distance_weight': 0.0111529223557, 'edge_usage_importance': 2.0590910938519, 'max_penalty': 1.0097276530193, 'min_penalty': 0.0359741233206, 'non_penalized_weight': 0.1937068090038, 'penalty_boost': 1.8537192725776, 'penalty_decay': 0.7424937626778, 'penalty_factor': 2.3013073955163, 'penalty_scale': 1.2224320044376, 'usage_weight': 0.5393776682261}
    scoringalg = OptimizedSynergisticDynamicPenaltyUpdate(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)