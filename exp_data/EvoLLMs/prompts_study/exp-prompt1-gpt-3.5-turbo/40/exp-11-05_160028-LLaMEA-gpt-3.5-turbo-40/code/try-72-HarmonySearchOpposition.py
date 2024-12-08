import numpy as np

class HarmonySearchOpposition:
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
            new_fitness = func(new_harmony)
            
            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                population[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
            
            # Generate Opposing Solution
            opposing_harmony = self.lower_bound + self.upper_bound - population[worst_idx]
            opposing_fitness = func(opposing_harmony)
            if opposing_fitness < fitness[worst_idx]:
                population[worst_idx] = opposing_harmony
                fitness[worst_idx] = opposing_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]