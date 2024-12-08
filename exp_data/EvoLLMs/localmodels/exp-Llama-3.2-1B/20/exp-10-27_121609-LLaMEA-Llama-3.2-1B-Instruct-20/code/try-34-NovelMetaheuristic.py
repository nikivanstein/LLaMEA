import numpy as np
import random

class NovelMetaheuristic:
    def __init__(self, budget, dim, mutation_rate, exploration_rate, bounds):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.exploration_rate = exploration_rate
        self.bounds = bounds
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            if random.random() < self.mutation_rate:
                i, j = random.sample(range(self.dim), 2)
                x = individual.copy()
                x[i], x[j] = x[j], x[i]
                return x

        def explore(individual):
            if random.random() < self.exploration_rate:
                i, j = random.sample(range(self.dim), 2)
                x = individual.copy()
                fitness = objective(x)
                return fitness

        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = explore(self.population[i])
                if fitness < self.fitnesses[i, individual[i]] + 1e-6:
                    self.fitnesses[i, individual[i]] = fitness
                    self.population[i] = mutate(individual[i])

        return self.fitnesses

# Example usage
func = lambda x: x**2
novel_metaheuristic = NovelMetaheuristic(100, 10, 0.1, 0.01, (-5.0, 5.0))
fitnesses = novel_metaheuristic(func)
print(fitnesses)