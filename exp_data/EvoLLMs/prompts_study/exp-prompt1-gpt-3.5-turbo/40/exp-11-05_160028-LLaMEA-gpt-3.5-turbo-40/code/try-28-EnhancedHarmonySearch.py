import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget - len(population)):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            
            if np.random.rand() < 0.1:  # Introduce mutation with 10% probability
                mutation = np.random.normal(0, 0.5, self.dim)  # Gaussian mutation with std 0.5
                new_harmony = np.clip(new_harmony + mutation, self.lower_bound, self.upper_bound)
                
            new_fitness = func(new_harmony)
            
            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                population[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]