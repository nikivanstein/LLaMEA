import numpy as np
from scipy.optimize import differential_evolution

class PyCromES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.tau = 0.9
        self.es_prob = 0.4

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
            population[idx] = np.random.uniform(-5.0, 5.0, size=(len(idx), self.dim))

            # Apply Evolutionary Strategies to refine the population
            es_idx = np.random.choice(self.population_size, size=int(self.population_size * self.es_prob), replace=False)
            for i in es_idx:
                # Select a random solution
                solution = population[i]

                # Generate a new solution
                new_solution = solution + np.random.normal(0, 1, size=self.dim)

                # Calculate the difference in fitness
                delta_fitness = func(new_solution) - func(solution)

                # Accept the new solution if it's better or with a certain probability
                if delta_fitness > 0 or np.random.rand() < np.exp(-(delta_fitness / 1000.0)):
                    population[i] = new_solution

        # Return the best solution
        return np.min(population, axis=0)