import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func, mutation_rate, local_search_rate):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.
            mutation_rate (float): The probability of mutation.
            local_search_rate (float): The probability of local search.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Generate a random point in the search space
            point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the current point
            value = func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = point

            # Perform mutation and local search
            if random.random() < mutation_rate:
                # Randomly swap two points in the search space
                point1, point2 = random.sample(self.search_space, 2)
                self.search_space[np.random.choice([0, 1], size=dim)] = point2

            if random.random() < local_search_rate:
                # Perform local search around the current point
                neighbors = self.get_neighbors(point, self.search_space)
                best_neighbor = None
                best_neighbor_value = float('-inf')
                for neighbor in neighbors:
                    value = func(neighbor)
                    if value > best_neighbor_value:
                        best_neighbor_value = value
                        best_neighbor = neighbor

                # Update the best point
                if best_neighbor_value > best_value:
                    best_value = best_neighbor_value
                    best_index = best_neighbor

        # Return the optimized value
        return best_value

    def get_neighbors(self, point, search_space):
        """
        Get the neighbors of a point in the search space.

        Args:
            point (float): The point to get neighbors for.
            search_space (numpy array): The search space.

        Returns:
            list: A list of neighbors.
        """
        neighbors = []
        for i in range(self.dim):
            neighbors.append(point + self.search_space[i])
            neighbors.append(point - self.search_space[i])
        return neighbors