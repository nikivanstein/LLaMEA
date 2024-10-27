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

        def mutate(x, p):
            r = np.random.rand(x.shape[0])
            return x + p * (x - x) * r

        def crossover(x1, x2, p):
            if np.random.rand() < p:
                return x1
            else:
                return x2

        def selection(x, k):
            return np.random.choice(x, k, replace=False)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        # Select the best individual based on the budget
        selected_individuals = np.argsort(-self.fitnesses, axis=0)
        selected_individuals = selection(selected_individuals, self.population_size // 2)

        # Perform mutation and crossover to refine the solution
        for i in range(self.population_size):
            if i < self.population_size // 2:
                x = mutation(self.population[i], 0.5)
            else:
                x = crossover(self.population[i], selected_individuals[i], 0.5)

            fitness = objective(x)
            if fitness < self.fitnesses[i, x] + 1e-6:
                self.fitnesses[i, x] = fitness
                self.population[i] = x

        return self.fitnesses

# One-line description with the main idea:
# Novel Hybrid Metaheuristic for Black Box Optimization
# 
# This algorithm combines NNEO (Neural Network-based Non-dominated Evolutionary Optimization) with mutation and crossover to refine the solution in each generation.