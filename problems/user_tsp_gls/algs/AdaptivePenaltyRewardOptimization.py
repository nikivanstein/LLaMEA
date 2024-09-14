import numpy as np

class AdaptivePenaltyRewardOptimization:
    def __init__(self, distance_weight, penalty_factor, usage_weight, reward_factor, penalty_decay, min_penalty, max_penalty, adaptive_factor, non_penalized_weight, penalty_scale, reward_threshold):
        self.distance_weight = distance_weight
        self.penalty_factor = penalty_factor
        self.usage_weight = usage_weight
        self.reward_factor = reward_factor
        self.penalty_decay = penalty_decay
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.adaptive_factor = adaptive_factor
        self.non_penalized_weight = non_penalized_weight
        self.penalty_scale = penalty_scale
        self.reward_threshold = reward_threshold

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        N = edge_distance.shape[0]
        updated_edge_distance = edge_distance.copy()

        # Apply penalties based on edge usage and distance
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

        # Apply adaptive penalties based on local optimal tour
        for idx in range(len(local_opt_tour) - 1):
            i, j = local_opt_tour[idx], local_opt_tour[idx + 1]
            adaptive_penalty = self.penalty_factor * (1 + self.adaptive_factor * edge_n_used[i, j])
            adaptive_penalty = np.clip(adaptive_penalty, self.min_penalty, self.max_penalty)
            updated_edge_distance[i, j] += adaptive_penalty * edge_distance[i, j]
            updated_edge_distance[j, i] += adaptive_penalty * edge_distance[i, j]

        # Decay or boost penalties based on a reward threshold
        for i in range(N):
            for j in range(i + 1, N):
                current_penalty = updated_edge_distance[i, j] - edge_distance[i, j]
                if current_penalty > self.reward_threshold:
                    updated_edge_distance[i, j] *= self.penalty_decay
                    updated_edge_distance[j, i] *= self.penalty_decay
                elif edge_n_used[i, j] == 0:
                    updated_edge_distance[i, j] -= self.reward_factor
                    updated_edge_distance[j, i] -= self.reward_factor

        # Scale the penalties
        updated_edge_distance *= self.penalty_scale
        updated_edge_distance = np.maximum(updated_edge_distance, 0)

        return updated_edge_distance

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # 1.850371707708594e-14
    config = {'adaptive_factor': 0.7717722453242, 'distance_weight': 1.3712778491209, 'max_penalty': 2.9307195618254, 'min_penalty': 0.3550375366202, 'non_penalized_weight': 1.0161133951766, 'penalty_decay': 0.9615175183891, 'penalty_factor': 1.0579198639018, 'penalty_scale': 0.9523749292262, 'reward_factor': 1.2055410277681, 'reward_threshold': 0.9823931117516, 'usage_weight': 0.1132653682633}
    scoringalg = AdaptivePenaltyRewardOptimization(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)