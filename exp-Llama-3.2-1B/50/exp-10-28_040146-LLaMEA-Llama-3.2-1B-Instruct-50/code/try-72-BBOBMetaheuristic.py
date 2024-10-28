# Code:
# import numpy as np
# import scipy.optimize as optimize
# import random

class BBOBMetaheuristic:
    """
    A metaheuristic algorithm for solving black box optimization problems.
    """

    def __init__(self, budget: int, dim: int):
        """
        Initialize the algorithm with a given budget and dimensionality.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func: Any, bounds: Dict[str, float] = None) -> Any:
        """
        Optimize the given black box function using the provided bounds.

        Args:
        func (Any): The black box function to optimize.
        bounds (Dict[str, float], optional): The bounds for the search space. Defaults to None.

        Returns:
        Any: The optimized function value.
        """
        # Create a copy of the function to avoid modifying the original function
        func_copy = func.copy()

        # Initialize the minimum and maximum function values
        min_func = np.inf
        max_func = -np.inf

        # Initialize the population size
        population_size = 100

        # Initialize the mutation rate
        mutation_rate = 0.01

        # Initialize the best solution found so far
        best_solution = None

        # Iterate over the population size
        for _ in range(population_size):
            # Initialize a new population with random solutions
            new_population = [np.random.uniform(-bounds["lower_bound"], bounds["upper_bound"], (self.dim,)) for _ in range(population_size)]

            # Evaluate the fitness of each individual in the new population
            fitnesses = [self.evaluate_fitness(individual, bounds, population_size) for individual in new_population]

            # Select the fittest individuals to reproduce
            fittest_individuals = sorted(zip(new_population, fitnesses), key=lambda x: x[1], reverse=True)[:self.budget]

            # Create a new population by breeding the fittest individuals
            new_population = [individual[0] for individual in fittest_individuals]

            # Evaluate the fitness of each individual in the new population
            fitnesses = [self.evaluate_fitness(individual, bounds, population_size) for individual in new_population]

            # Select the fittest individuals to mutate
            mutated_individuals = sorted(zip(new_population, fitnesses), key=lambda x: x[1], reverse=True)[:self.budget]

            # Create a new population by mutating the fittest individuals
            new_population = [individual[0] for individual in mutated_individuals]

            # Check if the new population is better than the best solution found so far
            if min_func > np.min([self.evaluate_fitness(individual, bounds, population_size) for individual in new_population]):
                # Update the best solution found so far
                best_solution = new_population[0]

                # Update the minimum and maximum function values
                min_func = np.min([self.evaluate_fitness(individual, bounds, population_size) for individual in new_population])
                max_func = np.max([self.evaluate_fitness(individual, bounds, population_size) for individual in new_population])

        # Return the optimized function value
        return min_func


# One-line description: A novel metaheuristic algorithm that uses a random search with bounds to optimize black box functions.
# Code: 
# ```python
# import numpy as np
# import scipy.optimize as optimize
#
# def bbobmetaheuristic(budget: int, dim: int) -> float:
#     return optimize.minimize(lambda x: x[0]**2 + x[1]**2, [1, 1], method="SLSQP", bounds=[(-5, 5), (-5, 5)]), budget=budget, dim=dim)