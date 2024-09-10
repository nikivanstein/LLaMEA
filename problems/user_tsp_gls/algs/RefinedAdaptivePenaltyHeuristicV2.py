import numpy as np

class RefinedAdaptivePenaltyHeuristicV2:
    def __init__(self, base_penalty, tour_penalty, usage_factor, gradient_factor, adaptive_threshold, decay_rate, penalty_reduction_factor, usage_penalty_weight, tour_specific_penalty_weight, distance_weight):
        self.base_penalty = base_penalty
        self.tour_penalty = tour_penalty
        self.usage_factor = usage_factor
        self.gradient_factor = gradient_factor
        self.adaptive_threshold = adaptive_threshold
        self.decay_rate = decay_rate
        self.penalty_reduction_factor = penalty_reduction_factor
        self.usage_penalty_weight = usage_penalty_weight
        self.tour_specific_penalty_weight = tour_specific_penalty_weight
        self.distance_weight = distance_weight

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        n = edge_distance.shape[0]
        updated_edge_distance = edge_distance.copy()

        # Compute base penalties on local optimal tour
        tour_length = len(local_opt_tour)
        for idx in range(tour_length - 1):
            u = local_opt_tour[idx]
            v = local_opt_tour[idx + 1]
            penalty = self.base_penalty + self.gradient_factor * (idx + 1)
            if edge_n_used[u, v] > self.adaptive_threshold:
                penalty *= self.penalty_reduction_factor
            updated_edge_distance[u, v] += penalty
            updated_edge_distance[v, u] += penalty

        # Apply penalties based on edge usage frequency
        for i in range(n):
            for j in range(n):
                if edge_n_used[i, j] > 0:
                    usage_penalty = self.usage_penalty_weight * (self.base_penalty + self.usage_factor * np.log1p(edge_n_used[i, j]) * np.exp(-self.decay_rate * edge_n_used[i, j]))
                    updated_edge_distance[i, j] += usage_penalty
                    updated_edge_distance[j, i] += usage_penalty

        # Apply tour-specific penalties
        for idx in range(tour_length - 1):
            u = local_opt_tour[idx]
            v = local_opt_tour[idx + 1]
            tour_specific_penalty = self.tour_specific_penalty_weight * (self.tour_penalty * np.exp(-self.decay_rate * idx))
            updated_edge_distance[u, v] += tour_specific_penalty
            updated_edge_distance[v, u] += tour_specific_penalty

        # Apply distance weighting
        for i in range(n):
            for j in range(n):
                updated_edge_distance[i, j] *= (1 + self.distance_weight * edge_distance[i, j])

        return updated_edge_distance

# Configuration Space:
configuration_space = {
    "base_penalty": (0.1, 10.0),
    "tour_penalty": (0.1, 10.0),
    "usage_factor": (0.1, 5.0),
    "gradient_factor": (0.1, 5.0),
    "adaptive_threshold": (0, 10),
    "decay_rate": (0.001, 1.0),
    "penalty_reduction_factor": (0.1, 1.0),
    "usage_penalty_weight": (0.1, 5.0),
    "tour_specific_penalty_weight": (0.1, 5.0),
    "distance_weight": (0.1, 10.0)
}

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # -3.700743415417189e-15
    config = {'adaptive_threshold': 10, 'base_penalty': 4.9383536778243, 'decay_rate': 0.0741478351077, 'distance_weight': 9.2851763950224, 'gradient_factor': 1.1761630481532, 'penalty_reduction_factor': 0.195895187676, 'tour_penalty': 6.9076440513301, 'tour_specific_penalty_weight': 0.3963130472514, 'usage_factor': 3.2336226416433, 'usage_penalty_weight': 1.6327726001031}
    scoringalg = RefinedAdaptivePenaltyHeuristicV2(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)