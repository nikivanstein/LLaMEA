import numpy as np
from scipy.optimize import differential_evolution

class ProbabilisticMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.tau = 0.9

    def __call__(self, func):
        if self.budget == 0:
            return None

        # Initialize the population with random solutions
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        for _ in range(self.budget):
            # Evaluate the fitness of each solution
            fitness = np.array([func(x) for x in population])

            # Select the fittest solutions
            fittest_idx = np.argsort(fitness)[:self.population_size // 2]
            population = population[fittest_idx]

            # Perform differential evolution to generate new solutions
            new_population = differential_evolution(lambda x: func(x), [(-5.0, 5.0) for _ in range(self.dim)], x0=population)

            # Update the population
            population = np.concatenate((population, new_population))

            # Apply mutation to some solutions
            idx = np.random.choice(self.population_size, size=int(self.population_size * self.mutation_rate), replace=False)
            mutated_population = population.copy()
            for i in idx:
                # Select a random dimension to mutate
                dim_idx = np.random.choice(self.dim)
                # Generate a new value with a probability of 0.4
                new_value = np.random.uniform(-5.0, 5.0)
                if np.random.rand() < 0.4:
                    mutated_population[i, dim_idx] = new_value
                else:
                    mutated_population[i, dim_idx] = population[i, dim_idx]

            population = mutated_population

        # Return the best solution
        return np.min(population, axis=0)