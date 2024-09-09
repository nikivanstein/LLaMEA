import numpy as np

class GradientAdaptivePenaltyOptimizerPlus:
    def __init__(self, base_penalty, tour_penalty, usage_factor, gradient_factor, decay_rate, edge_usage_decay, distance_weight, max_penalty_threshold, penalty_reduction_factor, usage_penalty_weight, tour_specific_penalty_weight, adaptive_decay_threshold, gradient_decay_factor, penalty_decay_rate, penalty_amplification_factor, neighborhood_penalty_weight):
        self.base_penalty = base_penalty
        self.tour_penalty = tour_penalty
        self.usage_factor = usage_factor
        self.gradient_factor = gradient_factor
        self.decay_rate = decay_rate
        self.edge_usage_decay = edge_usage_decay
        self.distance_weight = distance_weight
        self.max_penalty_threshold = max_penalty_threshold
        self.penalty_reduction_factor = penalty_reduction_factor
        self.usage_penalty_weight = usage_penalty_weight
        self.tour_specific_penalty_weight = tour_specific_penalty_weight
        self.adaptive_decay_threshold = adaptive_decay_threshold
        self.gradient_decay_factor = gradient_decay_factor
        self.penalty_decay_rate = penalty_decay_rate
        self.penalty_amplification_factor = penalty_amplification_factor
        self.neighborhood_penalty_weight = neighborhood_penalty_weight

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        n = edge_distance.shape[0]
        updated_edge_distance = edge_distance.copy()

        # Compute base penalties on local optimal tour
        tour_length = len(local_opt_tour)
        for idx in range(tour_length - 1):
            u = local_opt_tour[idx]
            v = local_opt_tour[idx + 1]
            penalty = self.base_penalty + self.gradient_factor * (idx + 1)
            if penalty > self.max_penalty_threshold:
                penalty *= self.penalty_reduction_factor
            if edge_n_used[u, v] > self.adaptive_decay_threshold:
                penalty *= self.gradient_decay_factor
            updated_edge_distance[u, v] += penalty
            updated_edge_distance[v, u] += penalty

        # Apply penalties based on edge usage frequency
        for i in range(n):
            for j in range(n):
                if edge_n_used[i, j] > 0:
                    usage_penalty = self.usage_penalty_weight * (self.base_penalty + self.usage_factor * edge_n_used[i, j] * np.exp(-self.edge_usage_decay * edge_n_used[i, j]))
                    if usage_penalty > self.max_penalty_threshold:
                        usage_penalty *= self.penalty_reduction_factor
                    updated_edge_distance[i, j] += usage_penalty
                    updated_edge_distance[j, i] += usage_penalty

        # Apply decaying tour-specific penalty
        for idx in range(tour_length - 1):
            u = local_opt_tour[idx]
            v = local_opt_tour[idx + 1]
            tour_specific_penalty = self.tour_specific_penalty_weight * (self.tour_penalty * np.exp(-self.decay_rate * idx))
            if tour_specific_penalty > self.max_penalty_threshold:
                tour_specific_penalty *= self.penalty_reduction_factor
            updated_edge_distance[u, v] += tour_specific_penalty
            updated_edge_distance[v, u] += tour_specific_penalty

        # Apply distance weighting
        for i in range(n):
            for j in range(n):
                updated_edge_distance[i, j] *= (1 + self.distance_weight * edge_distance[i, j])

        # Apply penalty decay and amplification to discourage frequent retracing of paths
        for i in range(n):
            for j in range(n):
                if edge_n_used[i, j] > self.adaptive_decay_threshold:
                    updated_edge_distance[i, j] *= (1 - self.penalty_decay_rate)
                else:
                    updated_edge_distance[i, j] *= (1 + self.penalty_amplification_factor)

        # Apply additional penalties to neighborhood edges
        for idx in range(tour_length - 1):
            u = local_opt_tour[idx]
            v = local_opt_tour[idx + 1]
            for neighbor in range(n):
                if neighbor != u and neighbor != v:
                    neighbor_penalty = self.neighborhood_penalty_weight * (self.base_penalty + self.gradient_factor * (idx + 1))
                    if neighbor_penalty > self.max_penalty_threshold:
                        neighbor_penalty *= self.penalty_reduction_factor
                    updated_edge_distance[u, neighbor] += neighbor_penalty
                    updated_edge_distance[v, neighbor] += neighbor_penalty
                    updated_edge_distance[neighbor, u] += neighbor_penalty
                    updated_edge_distance[neighbor, v] += neighbor_penalty

        return updated_edge_distance

# Configuration Space:
configuration_space = {
    "base_penalty": (0.1, 10.0),
    "tour_penalty": (0.1, 10.0),
    "usage_factor": (0.1, 5.0),
    "gradient_factor": (0.1, 5.0),
    "decay_rate": (0.001, 1.0),
    "edge_usage_decay": (0.001, 1.0),
    "distance_weight": (0.1, 10.0),
    "max_penalty_threshold": (5.0, 50.0),
    "penalty_reduction_factor": (0.1, 1.0),
    "usage_penalty_weight": (0.1, 5.0),
    "tour_specific_penalty_weight": (0.1, 5.0),
    "adaptive_decay_threshold": (0, 5),
    "gradient_decay_factor": (0.1, 1.0),
    "penalty_decay_rate": (0.001, 0.1),
    "penalty_amplification_factor": (0.001, 0.1),
    "neighborhood_penalty_weight": (0.1, 5.0)
}

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # -7.401486830834377e-15
    config = {'adaptive_decay_threshold': 5, 'base_penalty': 7.8775429635898, 'decay_rate': 0.9922692026343, 'distance_weight': 1.0772322043931, 'edge_usage_decay': 0.4677179658724, 'gradient_decay_factor': 0.1639449963734, 'gradient_factor': 2.8969402171445, 'max_penalty_threshold': 40.9543677566451, 'neighborhood_penalty_weight': 0.2442116558094, 'penalty_amplification_factor': 0.0824068689731, 'penalty_decay_rate': 0.0639694316203, 'penalty_reduction_factor': 0.601174440961, 'tour_penalty': 1.5942876711442, 'tour_specific_penalty_weight': 4.6739070791034, 'usage_factor': 3.761832169734, 'usage_penalty_weight': 0.9382231246655}
    scoringalg = GradientAdaptivePenaltyOptimizerPlus(**config)
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)