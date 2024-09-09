import numpy as np

class DynamicPenaltyGradientOptimizer:
    def __init__(self, base_penalty, usage_factor, tour_penalty, gradient_factor, decay_rate, edge_usage_decay, distance_weight, max_penalty_threshold, penalty_reduction_factor, usage_penalty_weight, tour_specific_penalty_weight):
        self.base_penalty = base_penalty
        self.usage_factor = usage_factor
        self.tour_penalty = tour_penalty
        self.gradient_factor = gradient_factor
        self.decay_rate = decay_rate
        self.edge_usage_decay = edge_usage_decay
        self.distance_weight = distance_weight
        self.max_penalty_threshold = max_penalty_threshold
        self.penalty_reduction_factor = penalty_reduction_factor
        self.usage_penalty_weight = usage_penalty_weight
        self.tour_specific_penalty_weight = tour_specific_penalty_weight

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        n = edge_distance.shape[0]
        updated_edge_distance = edge_distance.copy()
        
        # Apply penalty based on the number of times edges are used in the local optimal tour
        for i in range(len(local_opt_tour) - 1):
            u = local_opt_tour[i]
            v = local_opt_tour[i + 1]
            penalty = self.base_penalty + self.gradient_factor * (i + 1)
            if penalty > self.max_penalty_threshold:
                penalty *= self.penalty_reduction_factor
            updated_edge_distance[u, v] += penalty
            updated_edge_distance[v, u] += penalty
        
        # Apply additional penalties based on the overall usage of edges
        for i in range(n):
            for j in range(n):
                if edge_n_used[i, j] > 0:
                    usage_penalty = self.usage_penalty_weight * (self.base_penalty + self.usage_factor * edge_n_used[i, j] * np.exp(-self.edge_usage_decay * edge_n_used[i, j]))
                    if usage_penalty > self.max_penalty_threshold:
                        usage_penalty *= self.penalty_reduction_factor
                    updated_edge_distance[i, j] += usage_penalty
                    updated_edge_distance[j, i] += usage_penalty
        
        # Apply tour-specific penalty with decay to encourage diverse route exploration
        for i in range(len(local_opt_tour) - 1):
            u = local_opt_tour[i]
            v = local_opt_tour[i + 1]
            tour_specific_penalty = self.tour_specific_penalty_weight * (self.tour_penalty * np.exp(-self.decay_rate * i))
            if tour_specific_penalty > self.max_penalty_threshold:
                tour_specific_penalty *= self.penalty_reduction_factor
            updated_edge_distance[u, v] += tour_specific_penalty
            updated_edge_distance[v, u] += tour_specific_penalty

        # Apply distance weighting to amplify the penalties on longer edges
        for i in range(n):
            for j in range(n):
                updated_edge_distance[i, j] *= (1 + self.distance_weight * edge_distance[i, j])
        
        return updated_edge_distance

# Configuration Space
configuration_space = {
    "base_penalty": (0.1, 10.0),
    "usage_factor": (0.1, 5.0),
    "tour_penalty": (0.1, 10.0),
    "gradient_factor": (0.1, 5.0),
    "decay_rate": (0.001, 1.0),
    "edge_usage_decay": (0.001, 1.0),
    "distance_weight": (0.1, 10.0),
    "max_penalty_threshold": (5.0, 50.0),
    "penalty_reduction_factor": (0.1, 1.0),
    "usage_penalty_weight": (0.1, 5.0),
    "tour_specific_penalty_weight": (0.1, 5.0)
}

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # 1.850371707708594e-14
    config = {'base_penalty': 0.208140156142, 'decay_rate': 0.0030641310109, 'distance_weight': 8.1245988428622, 'edge_usage_decay': 0.2876661782624, 'gradient_factor': 3.7966517454799, 'max_penalty_threshold': 46.8305727744032, 'penalty_reduction_factor': 0.1005997845325, 'tour_penalty': 6.6278643180043, 'tour_specific_penalty_weight': 4.6615494208957, 'usage_factor': 4.7867055138923, 'usage_penalty_weight': 4.6430420841396}
    scoringalg = DynamicPenaltyGradientOptimizer(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)