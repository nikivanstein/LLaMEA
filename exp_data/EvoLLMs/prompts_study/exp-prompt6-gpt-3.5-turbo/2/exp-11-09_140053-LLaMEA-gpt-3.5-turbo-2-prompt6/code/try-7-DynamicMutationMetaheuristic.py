import numpy as np

class DynamicMutationMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.mutation_rate = 0.1

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = [func(ind) for ind in population]
        
        for _ in range(self.budget):
            offspring = []
            for i in range(self.pop_size):
                parent = population[i]
                child = parent + self.mutation_rate * np.random.randn(self.dim)
                if func(child) < fitness[i]:
                    population[i] = child
                    fitness[i] = func(child)
            self.mutation_rate *= 0.995  # Adaptive control of mutation rate
        return population[np.argmin(fitness)]