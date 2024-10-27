import numpy as np
from scipy.optimize import minimize

class NCE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.cluster_size = 5
        self.cluster_count = 0
        self.cluster_centers = []
        self.fitness_history = []
        self.expansion_rate = 0.05

    def __call__(self, func):
        # Initialize population with random niches
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        for i in range(self.population_size):
            self.cluster_count += 1
            if self.cluster_count > self.cluster_size:
                self.cluster_centers.append(self.cluster_size)
                self.cluster_count = 1
            population[i] = self.cluster_centers[np.random.randint(0, self.cluster_count)]

        # Evaluate population and store fitness history
        fitness_history = []
        for x in population:
            f = func(x)
            fitness_history.append(f)
            self.fitness_history.append(f)

        # Main loop
        for _ in range(self.budget):
            # Select fittest individuals
            fittest_individuals = np.argsort(fitness_history)[-self.population_size:]
            fittest_population = population[fittest_individuals]

            # Calculate niches
            niches = np.array_split(fittest_population, self.cluster_count)
            niches = np.array([np.mean(niche, axis=0) for niche in niches])

            # Update cluster centers
            self.cluster_centers = niches

            # Expand clusters
            new_population = []
            for i in range(self.population_size):
                if np.random.rand() < self.expansion_rate:
                    new_individual = np.random.uniform(-5.0, 5.0, size=self.dim)
                    new_individual = self.cluster_centers[np.random.randint(0, self.cluster_count)]
                    new_population.append(new_individual)
                else:
                    new_population.append(fittest_population[i])

            # Store new fitness history
            new_fitness_history = []
            for x in new_population:
                f = func(x)
                new_fitness_history.append(f)
                self.fitness_history.append(f)

            # Update population and fitness history
            population = new_population
            fitness_history = new_fitness_history

# Example usage
def func(x):
    return np.sum(x**2)

nce = NCE(budget=100, dim=10)
nce(func)