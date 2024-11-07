import numpy as np

class DynamicHarmonySearch:
    def __init__(self, budget, dim, pitch_adjust_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pitch_adjust_rate = pitch_adjust_rate

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
            else:
                pitch_adjustment = np.random.uniform(-self.pitch_adjust_rate, self.pitch_adjust_rate, self.dim)
                new_harmony = np.clip(population[worst_idx] + pitch_adjustment, self.lower_bound, self.upper_bound)
                new_fitness = func(new_harmony)
                if new_fitness < fitness[worst_idx]:
                    population[worst_idx] = new_harmony
                    fitness[worst_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]