import numpy as np
from scipy.optimize import minimize

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.population_size = 50
        self.mutation_rate = 0.1

    def __call__(self, func):
        def objective(x):
            return func(x)

        # Initialize population with random solutions
        population = np.random.uniform(self.search_space, size=(self.population_size, self.dim))

        for _ in range(self.budget):
            # Evaluate the objective function for each individual in the population
            evaluations = [objective(x) for x in population]

            # Select the fittest individuals
            fittest_indices = np.argsort(evaluations)[::-1][:self.population_size//2]
            fittest_individuals = population[fittest_indices]

            # Create a new population by mutating the fittest individuals
            new_population = np.array([x + np.random.normal(0, 1, self.dim) for x in fittest_individuals])

            # Evaluate the new population
            new_evaluations = [objective(x) for x in new_population]

            # Replace the old population with the new one
            population = np.concatenate((fittest_individuals, new_population), axis=0)

            # Apply mutation to the new population
            population = np.concatenate((population[:self.population_size//2], population[self.population_size//2:]), axis=0)
            population = np.random.uniform(self.search_space, size=(self.population_size, self.dim), axis=0) + np.random.normal(0, 1, self.dim, size=(self.population_size, self.dim))

            # Evaluate the new population
            new_evaluations = [objective(x) for x in population]

            # Replace the old population with the new one
            population = np.concatenate((population[:self.population_size//2], new_population), axis=0)

        # Return the fittest individual in the final population
        return population[np.argmax(evaluations)][self.dim//2]

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

evolutionary_algorithm = EvolutionaryAlgorithm(1000, 2)  # 1000 function evaluations, 2 dimensions
print(evolutionary_algorithm(test_function))  # prints a random value between -10 and 10