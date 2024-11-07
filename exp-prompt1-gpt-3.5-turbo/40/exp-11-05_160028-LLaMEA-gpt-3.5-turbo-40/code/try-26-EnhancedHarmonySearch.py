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
            new_fitness = func(new_harmony)
            
            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                population[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
            
            # Opposition-based Learning
            opp_harmony = self.lower_bound + self.upper_bound - population
            opp_fitness = np.array([func(x) for x in opp_harmony])
            
            best_opp_idx = np.argmin(opp_fitness)
            if opp_fitness[best_opp_idx] < fitness[worst_idx]:
                population[worst_idx] = opp_harmony[best_opp_idx]
                fitness[worst_idx] = opp_fitness[best_opp_idx]
        
        best_idx = np.argmin(fitness)
        return population[best_idx]