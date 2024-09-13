import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Create a copy of the edge distance and edge used matrices
    updated_edge_distance = np.copy(edge_distance)
    updated_edge_n_used = np.copy(edge_n_used)
    
    # Get the number of nodes in the tour
    num_nodes = len(local_opt_tour)
    total_permutations = num_nodes * (num_nodes - 1)
    
    # Iterate over each node in the tour
    for i in range(num_nodes):
        # Get the current node ID and the next node ID
        current_node = local_opt_tour[i]
        next_node = local_opt_tour[(i + 1) % num_nodes]  # Wrap around to the first node if at the last node
        
        # Compute the score for updating the edge distance
        score = (total_permutations - edge_n_used[current_node, next_node]) / (edge_n_used[current_node, next_node] + 1)
        
        # Update the edge distance between the current and next node
        updated_edge_distance[current_node, next_node] = edge_distance[current_node, next_node] + score
        
        # Update the edge distance between the next and current node
        updated_edge_distance[next_node, current_node] = edge_distance[next_node, current_node] + score
        
        # Update the number of times the edge is used
        updated_edge_n_used[current_node, next_node] += 1
        updated_edge_n_used[next_node, current_node] += 1
    
    return updated_edge_distance