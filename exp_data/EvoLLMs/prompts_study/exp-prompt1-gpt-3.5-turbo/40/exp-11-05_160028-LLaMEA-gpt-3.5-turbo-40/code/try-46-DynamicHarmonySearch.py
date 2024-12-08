import numpy as np

class DynamicHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.bandwidth = np.ones(dim) * 0.2

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget - len(population)):
            new_harmony = np.random.uniform(population.mean(axis=0) - self.bandwidth, population.mean(axis=0) + self.bandwidth)
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_fitness = func(new_harmony)
            
            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                population[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
                self.bandwidth *= 0.99  # Decrease bandwidth to focus on promising regions
            else:
                self.bandwidth *= 1.01  # Increase bandwidth to explore diverse regions
        
        best_idx = np.argmin(fitness)
        return population[best_idx]