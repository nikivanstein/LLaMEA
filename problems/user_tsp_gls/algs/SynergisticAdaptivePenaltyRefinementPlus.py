import numpy as np

class SynergisticAdaptivePenaltyRefinementPlus:
    def __init__(self, distance_weight, penalty_factor, usage_weight, penalty_decay, penalty_boost, min_penalty, max_penalty, adaptive_factor, adaptive_threshold):
        self.distance_weight = distance_weight
        self.penalty_factor = penalty_factor
        self.usage_weight = usage_weight
        self.penalty_decay = penalty_decay
        self.penalty_boost = penalty_boost
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.adaptive_factor = adaptive_factor
        self.adaptive_threshold = adaptive_threshold

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        N = edge_distance.shape[0]
        updated_edge_distance = edge_distance.copy()

        # Synergistic penalty adjustments
        for i in range(N):
            for j in range(i + 1, N):
                used_penalty = self.usage_weight * edge_n_used[i, j]
                distance_penalty = self.distance_weight * edge_distance[i, j]
                total_penalty = self.penalty_factor * (used_penalty + distance_penalty)
                total_penalty = np.clip(total_penalty, self.min_penalty, self.max_penalty)

                updated_edge_distance[i, j] += total_penalty
                updated_edge_distance[j, i] += total_penalty

        # Adaptive penalties for local optimal tour
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
    config = {'adaptive_factor': 1.0641410596094, 'adaptive_threshold': 0.9310466933181, 'distance_weight': 0.9289766329857, 'max_penalty': 2.4180365028989, 'min_penalty': 0.4475568443419, 'penalty_boost': 1.0384246004902, 'penalty_decay': 0.5487752416234, 'penalty_factor': 1.6573643291604, 'usage_weight': 0.1101292632286}
    scoringalg = SynergisticAdaptivePenaltyRefinementPlus(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)