import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    edge_count = np.zeros_like(edge_distance)

    for i in range(len(local_opt_tour) - 1):
        start = local_opt_tour[i]
        end = local_opt_tour[i + 1]
        edge_count[start][end] += 1
        edge_count[end][start] += 1

    edge_count[edge_n_used == 0] = 1  # simplified to avoid overfitting

    updated_edge_distance *= (1 + edge_count / edge_n_used)

    return updated_edge_distance