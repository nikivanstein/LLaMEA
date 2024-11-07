import numpy as np

class DynamicHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim, bandwidth=0.5):
        super().__init__(budget, dim)
        self.bandwidth = bandwidth

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget - len(population)):
            new_harmony = np.copy(population[np.random.randint(0, len(population))])
            for i in range(self.dim):
                if np.random.rand() < self.bandwidth:
                    new_harmony[i] = np.clip(new_harmony[i] + np.random.normal(0, 1), self.lower_bound, self.upper_bound)
            
            new_fitness = func(new_harmony)
            
            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                population[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]