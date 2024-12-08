import numpy as np
from scipy.optimize import minimize

class NCE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.niche_size = 5
        self.cluster_size = 10
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitness_history = []

    def __call__(self, func):
        # Initialize population with random niches
        for i in range(self.population_size):
            self.population[i] = np.random.uniform(-5.0, 5.0, size=self.dim)

        # Evaluate population and store fitness history
        self.fitness_history = [func(x) for x in self.population]

        # Main loop
        for _ in range(self.budget):
            # Select fittest individuals
            fittest_individuals = np.argsort(self.fitness_history)[-self.population_size:]
            fittest_population = self.population[fittest_individuals]

            # Cluster fittest individuals
            clusters = []
            cluster = []
            for individual in fittest_population:
                if not cluster or np.linalg.norm(individual - cluster[0]) > 1.0:
                    cluster = [individual]
                else:
                    cluster.append(individual)
                if len(cluster) == self.cluster_size:
                    clusters.append(cluster)
                    cluster = []
            if cluster:
                clusters.append(cluster)

            # Update population and fitness history
            new_population = []
            for cluster in clusters:
                new_cluster = np.mean(cluster, axis=0)
                new_population.extend([new_cluster] * self.population_size // len(clusters))
            self.population = new_population
            self.fitness_history = [func(x) for x in self.population]

            # Explore new niches
            for i in range(self.population_size):
                if np.random.rand() < 0.05:
                    self.population[i] = self.population[i] + np.random.uniform(-1.0, 1.0, size=self.dim)

# Example usage
def func(x):
    return np.sum(x**2)

nce = NCE(budget=100, dim=10)
nce(func)