import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def adaptive_black_box(self, func, budget, dim, max_iter=100, tol=1e-6):
        """Adaptive Black Box Optimization Algorithm"""
        # Initialize the population size
        pop_size = 100

        # Initialize the population
        population = self.generate_population(pop_size, dim, budget, dim)

        # Run the optimization algorithm
        for _ in range(max_iter):
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=lambda x: x.fitness, reverse=True)[:self.budget]

            # Select a subset of individuals based on the probability of refinement
            idx = np.random.choice(len(fittest_individuals), size=int(pop_size * 0.35), replace=False)
            fittest_individuals = [fittest_individuals[i] for i in idx]

            # Perform the refinement step
            for individual in fittest_individuals:
                # Evaluate the function at the current individual
                func_value = func(individual)

                # Refine the individual using the following steps:
                # 1. Search for the minimum and maximum of the function's values
                # 2. Update the individual using the following formula:
                #   new_individual = min(max(individual, func_value), max(individual, func_value + 0.1)) + 0.05
                # 3. Evaluate the function at the new individual
                # 4. Update the individual if it is better than the current best individual
                new_individual = individual.copy()
                for i in range(dim):
                    new_individual[i] = (new_individual[i] + func_value - new_individual[i]) / 2
                func_value = func(new_individual)
                if func_value < individual.fitness:
                    new_individual = func(new_individual)

                # Add the new individual to the population
                population.append(new_individual)

            # Replace the fittest individuals with the new individuals
            fittest_individuals = sorted(population, key=lambda x: x.fitness, reverse=True)
            population = fittest_individuals[:self.budget]

        # Return the fittest individual
        return population[0]

    def generate_population(self, pop_size, dim, budget, dim):
        """Generate a population of individuals"""
        # Initialize the population with random values
        population = np.random.uniform(-5.0, 5.0, size=(pop_size, dim))
        return population

# Description: Adaptive Black Box Optimization Algorithm
# Code: 
# ```python
# AdaptiveBlackBoxOptimizer(budget, dim).adaptive_black_box(func, budget, dim)