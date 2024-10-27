import numpy as np
import random
import copy

class MetaMetaHeuristics:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-metaheuristic algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.population = []

    def __call__(self, func):
        """
        Optimize the black box function `func` using meta-metaheuristic.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population with random parameter values
        self.population = [copy.deepcopy(func) for _ in range(self.budget)]

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.population[_ % self.budget] + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.population[_ % self.budget] += self.noise * np.random.normal(0, 1, self.dim)

        # Select the best individual based on the fitness
        self.population = sorted(self.population, key=lambda x: x.fun, reverse=True)[:self.budget // 2]

        # Return the optimized parameter values and the objective function value
        return self.population[0], func(self.population[0] + self.noise * np.random.normal(0, 1, self.dim))

    def mutate(self, individual):
        """
        Mutate an individual to introduce new strategies.

        Args:
            individual (individual): The individual to mutate.

        Returns:
            mutated_individual: The mutated individual.
        """
        # Introduce new strategies by adding or removing parameters
        mutated_individual = copy.deepcopy(individual)
        mutated_individual.params.append(random.uniform(-5.0, 5.0))
        return mutated_individual

# Example usage:
meta_metaheuristics = MetaMetaHeuristics(100, 10)
func = lambda x: x**2
optimal_individual, optimal_function_value = meta_metaheuristics(__call__(func))
print("Optimal Individual:", optimal_individual)
print("Optimal Function Value:", optimal_function_value)

# Select the best individual based on the fitness
selected_individual = meta_metaheuristics.population[0]
selected_individual = meta_metaheuristics.mutate(selected_individual)
print("Selected Individual:", selected_individual)
print("Selected Individual's Fitness:", meta_metaheuristics(__call__(selected_individual)))