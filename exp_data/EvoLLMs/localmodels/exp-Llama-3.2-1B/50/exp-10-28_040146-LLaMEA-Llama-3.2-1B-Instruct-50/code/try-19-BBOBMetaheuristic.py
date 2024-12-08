# Description: A novel metaheuristic algorithm that uses a random search with bounds to optimize black box functions.
# Code: 
# ```python
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any

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

    def __call__(self, func: Any, bounds: Dict[str, float] = None, population_size: int = 100, mutation_rate: float = 0.01) -> Any:
        """
        Optimize the given black box function using the provided bounds and population size.

        Args:
        func (Any): The black box function to optimize.
        bounds (Dict[str, float], optional): The bounds for the search space. Defaults to None.
        population_size (int, optional): The size of the population. Defaults to 100.
        mutation_rate (float, optional): The rate at which individuals are mutated. Defaults to 0.01.

        Returns:
        Any: The optimized function value.
        """
        # Initialize the population with random individuals
        population = np.random.uniform(-bounds["lower_bound"], bounds["upper_bound"], (self.population_size, self.dim), dtype=float)

        # Define the fitness function
        def fitness(individual: np.ndarray) -> float:
            # Evaluate the function using the individual
            func_value = func(individual)

            # Calculate the fitness
            fitness = np.exp(-func_value**2)

            return fitness

        # Evaluate the fitness of each individual in the population
        fitness_values = fitness(population)

        # Select the fittest individuals
        fittest_individuals = np.argsort(fitness_values)[-self.population_size:]

        # Create a copy of the population to avoid modifying the original population
        new_population = population[fittest_individuals]

        # Perform mutation on the new population
        for _ in range(self.budget):
            # Select two random individuals from the new population
            individual1 = new_population[np.random.choice(new_population.shape[0], 1)]
            individual2 = new_population[np.random.choice(new_population.shape[0], 1)]

            # Calculate the mutation probabilities
            mutation1 = np.random.rand()
            mutation2 = np.random.rand()

            # Perform mutation on individual1
            if mutation1 < mutation_rate:
                new_individual1 = np.random.uniform(-bounds["lower_bound"], bounds["upper_bound"], (self.dim,))

                # Perform mutation on individual2
                if mutation2 < mutation_rate:
                    new_individual2 = np.random.uniform(-bounds["lower_bound"], bounds["upper_bound"], (self.dim,))

            # Update the new population
            new_population[fittest_individuals] = [individual1, individual2]

        # Return the optimized function value
        return fitness(new_population[fittest_individuals].mean(axis=0))


# One-line description: A novel metaheuristic algorithm that uses a random search with bounds to optimize black box functions.