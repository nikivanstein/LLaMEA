# Description: Evolutionary Optimization of Black Box Functions
# Code: 
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any

class BlackBoxOptimizer:
    def __init__(self, budget: int, dim: int, func: Dict[str, float]) -> None:
        """
        Initialize the BlackBoxOptimizer with a given budget, dimension, and a black box function.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.
        """
        self.budget = budget
        self.dim = dim
        self.func = func

    def __call__(self, func: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize the black box function using an evolutionary strategy.

        Args:
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.

        Returns:
        Dict[str, float]: The optimized function values.
        """
        # Initialize the search space with random values
        x = np.random.uniform(-5.0, 5.0, self.dim)

        # Define the objective function to minimize (negative of the original function)
        def objective(x: np.ndarray) -> float:
            return -np.sum(self.func.values(x))

        # Define the bounds for the search space
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]

        # Initialize the population with random individuals
        population = [x for _ in range(100)]

        # Run the evolution process
        for _ in range(100):
            # Select the fittest individuals
            fittest = population[np.argsort(self.func.values(population))[:self.budget]]

            # Create a new generation by mutating the fittest individuals
            new_population = []
            for _ in range(self.budget):
                # Select a random individual from the fittest generation
                individual = fittest[np.random.randint(0, len(fittest))]

                # Mutate the individual using the selected strategy
                if np.random.rand() < 0.45:
                    # Randomly change a random variable
                    individual = np.random.uniform(-5.0, 5.0, self.dim)
                    individual[0] += np.random.uniform(-1.0, 1.0)
                    individual[1] += np.random.uniform(-1.0, 1.0)

                new_population.append(individual)

            # Replace the old population with the new one
            population = new_population

        # Evaluate the fittest individual in the new population
        updated_individual = population[np.argsort(self.func.values(population))[:self.budget]].pop()

        # Return the updated individual
        return {k: -v for k, v in updated_individual.items()}

# Description: Evolutionary Optimization of Black Box Functions
# Code: 