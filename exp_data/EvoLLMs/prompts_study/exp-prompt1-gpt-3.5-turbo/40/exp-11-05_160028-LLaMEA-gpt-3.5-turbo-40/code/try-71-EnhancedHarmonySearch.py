import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.p_rate = 0.5  # Probability of applying pitch adjustment

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget - len(population)):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            new_fitness = func(new_harmony)
            
            # Dynamic pitch adjustment
            if np.random.rand() < self.p_rate:
                p = np.random.uniform(0, 1, self.dim)
                new_harmony = new_harmony + p * (population[np.random.randint(0, len(population))] - new_harmony)
                new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
                new_fitness = func(new_harmony)
            
            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                population[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]