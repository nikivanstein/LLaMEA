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

def mutate(individual):
    """
    Randomly mutate an individual by swapping two random points.

    Args:
        individual (List[float]): The individual to mutate.

    Returns:
        List[float]: The mutated individual.
    """
    if len(individual) < 2:
        return individual

    # Randomly select two points
    idx1, idx2 = random.sample(range(len(individual)), 2)

    # Swap the points
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    return individual

def adapt_search_space(individual, best_value, budget):
    """
    Adapt the search space based on the best value found so far.

    Args:
        individual (List[float]): The individual to adapt.
        best_value (float): The best value found so far.
        budget (int): The remaining budget.

    Returns:
        List[float]: The adapted individual.
    """
    # Calculate the proportion of the search space that has been explored
    proportion = (best_value - individual) / (5.0 - individual)

    # Adapt the search space
    adapted_space = [i * proportion for i in individual]

    # Limit the search space to the range [-5.0, 5.0]
    adapted_space = np.clip(adapted_space, -5.0, 5.0)

    return adapted_space

def bbo(func, budget, dim):
    """
    Optimize the black box function using the BlackBoxOptimizer.

    Args:
        func (callable): The black box function to optimize.
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.

    Returns:
        float: The optimized value of the function.
    """
    optimizer = BlackBoxOptimizer(budget, dim)
    for _ in range(budget):
        individual = optimizer(func)
        best_value = individual

        # Adapt the search space based on the best value found so far
        adapted_individual = adapt_search_space(individual, best_value, budget)

        # Mutate the individual
        mutated_individual = mutate(adapted_individual)

        # Evaluate the function at the mutated individual
        value = func(mutated_individual)

        # If the current value is better than the best value found so far,
        # update the best value and its corresponding index
        if value > best_value:
            best_value = value
            best_index = mutated_individual

    return best_value

# Example usage:
def sphere(func, budget):
    return bbo(func, budget, 5)

# Evaluate the function
best_value = sphere(sphere, 100)
print("Best value:", best_value)