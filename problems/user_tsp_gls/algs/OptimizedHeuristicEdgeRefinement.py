import numpy as np

class OptimizedHeuristicEdgeRefinement:
    def __init__(self, 
                 distance_weight, 
                 penalty_factor, 
                 usage_weight, 
                 reward_factor, 
                 min_penalty, 
                 max_penalty, 
                 adaptive_factor, 
                 non_penalized_weight, 
                 penalty_scale, 
                 reward_threshold, 
                 adaptive_reward_factor, 
                 max_reward):
        self.distance_weight = distance_weight
        self.penalty_factor = penalty_factor
        self.usage_weight = usage_weight
        self.reward_factor = reward_factor
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.adaptive_factor = adaptive_factor
        self.non_penalized_weight = non_penalized_weight
        self.penalty_scale = penalty_scale
        self.reward_threshold = reward_threshold
        self.adaptive_reward_factor = adaptive_reward_factor
        self.max_reward = max_reward

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
                    updated_edge_distance[i, j] *= (1 - self.penalty_scale)
                    updated_edge_distance[j, i] *= (1 - self.penalty_scale)
                elif edge_n_used[i, j] == 0:
                    adaptive_reward = self.reward_factor * (1 + self.adaptive_reward_factor * (self.max_reward - edge_distance[i, j]))
                    updated_edge_distance[i, j] = max(updated_edge_distance[i, j] - adaptive_reward, 0)
                    updated_edge_distance[j, i] = max(updated_edge_distance[j, i] - adaptive_reward, 0)

        return updated_edge_distance

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # 1.850371707708594e-14
    config = {'adaptive_factor': 1.6221888882946, 'adaptive_reward_factor': 3.6309064301196, 'distance_weight': 4.3386589319073, 'max_penalty': 1.3277930486947, 'max_reward': 5.5147998929024, 'min_penalty': 1.9262954203784, 'non_penalized_weight': 2.822219046047, 'penalty_factor': 3.835526753217, 'penalty_scale': 0.2064996041078, 'reward_factor': 2.7715013733786, 'reward_threshold': 0.6121571401134, 'usage_weight': 2.2428382257}
    scoringalg = OptimizedHeuristicEdgeRefinement(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)