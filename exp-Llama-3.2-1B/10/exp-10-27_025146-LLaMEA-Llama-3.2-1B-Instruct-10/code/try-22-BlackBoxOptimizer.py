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

    def __call__(self, func):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.

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

        # Return the optimized value
        return best_value

class BlackBoxOptimizerMetaheuristic(BlackBoxOptimizer):
    def __init__(self, budget, dim, strategy):
        """
        Initialize the BlackBoxOptimizerMetaheuristic with a budget, dimensionality, and strategy.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
            strategy (list): A list of tuples, where each tuple contains the mutation rate, crossover rate, and replacement rate.
        """
        super().__init__(budget, dim)
        self.strategy = strategy

    def __call__(self, func):
        """
        Optimize the black box function using the BlackBoxOptimizerMetaheuristic.

        Args:
            func (callable): The black box function to optimize.

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

            # Apply the mutation strategy
            mutated_point = self.mutate(point, self.strategy)

            # Evaluate the function at the mutated point
            mutated_value = func(mutated_point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if mutated_value > best_value:
                best_value = mutated_value
                best_index = mutated_point

        # Return the optimized value
        return best_value

def mutate(individual, strategy):
    """
    Apply the mutation strategy to the individual.

    Args:
        individual (list): The individual to mutate.
        strategy (list): A list of tuples, where each tuple contains the mutation rate, crossover rate, and replacement rate.

    Returns:
        list: The mutated individual.
    """
    mutation_rate, crossover_rate, replacement_rate = strategy
    mutated_individual = individual.copy()
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            mutated_individual[i] += np.random.uniform(-1, 1)
    # Crossover with another individual
    if np.random.rand() < crossover_rate:
        mutated_individual = crossover(mutated_individual, individual)
    # Replace one random individual with another individual
    if np.random.rand() < replacement_rate:
        mutated_individual = replace(mutated_individual, individual)
    return mutated_individual

def crossover(individual1, individual2):
    """
    Perform crossover between two individuals.

    Args:
        individual1 (list): The first individual.
        individual2 (list): The second individual.

    Returns:
        list: The offspring individual.
    """
    crossover_point = np.random.randint(0, len(individual1))
    offspring = individual1[:crossover_point] + individual2[crossover_point:]
    return offspring

def replace(individual1, individual2):
    """
    Replace one individual with another individual.

    Args:
        individual1 (list): The first individual.
        individual2 (list): The second individual.

    Returns:
        list: The offspring individual.
    """
    if np.random.rand() < replacement_rate:
        individual1 = individual2
    return individual1

# Example usage
def func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizerMetaheuristic(100, 5, [(0.1, 0.5, 0.3)])
print(optimizer(func))  # Output: Optimized value