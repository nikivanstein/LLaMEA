# Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
import copy

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

def novel_metaheuristic(func, budget, dim):
    """
    Novel Metaheuristic Algorithm for Black Box Optimization

    Description: This algorithm uses a novel metaheuristic approach to optimize black box functions.
    The algorithm starts by generating a random solution and then iteratively applies a series of mutations to improve the solution.
    The mutation strategy is based on the idea of "reducing the problem size" by removing a random subset of dimensions.
    The algorithm continues to evolve the solution until it reaches a specified budget.

    Code: 
    ```python
# Initialize the BlackBoxOptimizer with a budget and dimensionality
optimizer = BlackBoxOptimizer(budget, dim)

# Initialize the current solution
current_solution = None

# Initialize the mutation counter
mutation_counter = 0

# Iterate until the budget is reached
while current_solution is None:
    # Generate a random solution
    if current_solution is None:
        current_solution = np.random.randint(0, dim)
    else:
        current_solution = np.random.choice(current_solution, size=dim)

    # Evaluate the function at the current solution
    value = func(current_solution)

    # If the current value is better than the best value found so far,
    # update the best value and its corresponding index
    if value > best_value:
        best_value = value
        best_index = current_solution

    # Increment the mutation counter
    mutation_counter += 1

    # If the mutation counter exceeds the budget, stop evolving
    if mutation_counter >= budget:
        break

    # Apply a mutation to the current solution
    mutation = np.random.randint(0, dim)
    current_solution[mutation] = random.uniform(-5.0, 5.0)

# Return the optimized value
return best_value

# Example usage
def sphere(x):
    return np.sum(x**2)

best_value = novel_metaheuristic(sphere, 100, 5)
print("Best value:", best_value)