import numpy as np

class ImprovedDynamicMutationMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.mutation_rate = np.full(self.pop_size, 0.1)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = [func(ind) for ind in population]
        
        for _ in range(self.budget):
            offspring = []
            for i in range(self.pop_size):
                parent = population[i]
                child = parent + self.mutation_rate[i] * np.random.randn(self.dim)
                if func(child) < fitness[i]:
                    population[i] = child
                    fitness[i] = func(child)
                    self.mutation_rate[i] *= 1.02  # Adaptive control of mutation rate based on success
                else:
                    self.mutation_rate[i] *= 0.98  # Adjusting mutation rate for unsuccessful individuals
        return population[np.argmin(fitness)]