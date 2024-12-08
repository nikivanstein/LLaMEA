import numpy as np

class ImprovedDynamicPopulationSizeAdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.step_size = 0.1

    def levy_flight(self, alpha=0.1):
        return np.random.standard_cauchy(self.dim) * self.step_size / np.power(np.abs(np.random.normal()) + 1e-10, 1.0 / alpha)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = [func(ind) for ind in population]
        
        for _ in range(self.budget):
            offspring = []
            for i in range(self.pop_size):
                parent = population[i]
                child = parent + self.levy_flight()
                if func(child) < fitness[i]:
                    population[i] = child
                    fitness[i] = func(child)
                else:
                    population[i] = parent
        return population[np.argmin(fitness)]