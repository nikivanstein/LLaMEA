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

def novel_metaheuristic_optimizer(budget, dim):
    """
    Novel Metaheuristic Algorithm for Black Box Optimization.

    The algorithm uses a variant of the simulated annealing technique, 
    with a probability of 0.1 to change the individual lines of the strategy.
    """
    # Initialize the best solution and its corresponding fitness
    best_solution = None
    best_fitness = float('-inf')

    # Initialize the temperature and the cooling rate
    temperature = 1000.0
    cooling_rate = 0.99

    # Initialize the current solution and its fitness
    current_solution = None
    current_fitness = float('-inf')

    # Perform the specified number of function evaluations
    for _ in range(budget):
        # Generate a random solution in the search space
        current_solution = np.random.choice(self.search_space, dim)

        # Evaluate the current solution
        current_fitness = func(current_solution)

        # If the current fitness is better than the best fitness found so far,
        # update the best fitness and the best solution
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = current_solution

        # If the current fitness is worse than the best fitness found so far,
        # perform a simulated annealing step
        if current_fitness < best_fitness:
            # Generate a new solution with a probability of 0.1
            new_solution = current_solution + np.random.normal(0.0, 1.0, dim)

            # Evaluate the new solution
            new_fitness = func(new_solution)

            # If the new solution is better than the best solution found so far,
            # update the best solution and its corresponding fitness
            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_solution = new_solution

        # If the current fitness is equal to the best fitness found so far,
        # decrease the temperature and increase the cooling rate
        if current_fitness == best_fitness:
            temperature *= cooling_rate
            cooling_rate *= 0.99

    # Return the best solution and its corresponding fitness
    return best_solution, best_fitness

# Example usage:
func = lambda x: np.sin(x)
best_solution, best_fitness = novel_metaheuristic_optimizer(100, 5)
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")