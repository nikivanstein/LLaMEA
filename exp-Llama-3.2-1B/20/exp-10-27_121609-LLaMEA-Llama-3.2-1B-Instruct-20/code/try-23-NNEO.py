import numpy as np

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            new_individual = individual.copy()
            if np.random.rand() < 0.2:  # 20% chance of mutation
                new_individual[np.random.randint(0, self.dim)] = np.random.uniform(-5.0, 5.0)
            return new_individual

        def crossover(parent1, parent2):
            child = parent1.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:  # 50% chance of crossover
                    child[i] = parent2[i]
            return child

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

# Novel Metaheuristic Algorithm: "Dynamic Perturbation and Evolutionary Crossover"
# Code: 