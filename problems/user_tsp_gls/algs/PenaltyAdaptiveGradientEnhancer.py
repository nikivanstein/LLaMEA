import numpy as np

class PenaltyAdaptiveGradientEnhancer:
    def __init__(self, base_penalty, usage_factor, tour_penalty, gradient_factor, decay_rate, edge_usage_decay, distance_weight):
        self.base_penalty = base_penalty
        self.usage_factor = usage_factor
        self.tour_penalty = tour_penalty
        self.gradient_factor = gradient_factor
        self.decay_rate = decay_rate
        self.edge_usage_decay = edge_usage_decay
        self.distance_weight = distance_weight

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        n = edge_distance.shape[0]
        updated_edge_distance = edge_distance.copy()
        
        # Apply penalty based on the number of times edges are used in the local optimal tour
        for i in range(len(local_opt_tour) - 1):
            u = local_opt_tour[i]
            v = local_opt_tour[i + 1]
            penalty = self.base_penalty + self.gradient_factor * (i + 1)
            updated_edge_distance[u, v] += penalty
            updated_edge_distance[v, u] += penalty
        
        # Apply additional penalties based on the overall usage of edges
        for i in range(n):
            for j in range(n):
                if edge_n_used[i, j] > 0:
                    usage_penalty = self.base_penalty + self.usage_factor * edge_n_used[i, j] * np.exp(-self.edge_usage_decay * edge_n_used[i, j])
                    updated_edge_distance[i, j] += usage_penalty
                    updated_edge_distance[j, i] += usage_penalty
        
        # Apply tour-specific penalty with decay to encourage diverse route exploration
        for i in range(len(local_opt_tour) - 1):
            u = local_opt_tour[i]
            v = local_opt_tour[i + 1]
            tour_specific_penalty = self.tour_penalty * np.exp(-self.decay_rate * i)
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
    "distance_weight": (0.1, 10.0)
}

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # -1.1102230246251565e-14
    config = {'base_penalty': 1.7288726901988, 'decay_rate': 0.4811401685523, 'distance_weight': 7.6773775274509, 'edge_usage_decay': 0.0346165347965, 'gradient_factor': 1.3984572017284, 'tour_penalty': 6.4375987275872, 'usage_factor': 3.1143703227774}
    scoringalg = PenaltyAdaptiveGradientEnhancer(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)