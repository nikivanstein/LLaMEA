import numpy as np

class AdaptiveMutationMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.initial_mutation_rate = 0.1
        self.min_mutation_rate = 0.01

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = [func(ind) for ind in population]
        
        for _ in range(self.budget):
            offspring = []
            for i in range(self.pop_size):
                parent = population[i]
                mutation_rate = np.maximum(self.initial_mutation_rate / np.sqrt(1 + np.log(1 + func(parent))), self.min_mutation_rate)
                child = parent + mutation_rate * np.random.randn(self.dim)
                if func(child) < fitness[i]:
                    population[i] = child
                    fitness[i] = func(child)
            self.initial_mutation_rate *= 0.995  # Adaptive control of mutation rate
        return population[np.argmin(fitness)]