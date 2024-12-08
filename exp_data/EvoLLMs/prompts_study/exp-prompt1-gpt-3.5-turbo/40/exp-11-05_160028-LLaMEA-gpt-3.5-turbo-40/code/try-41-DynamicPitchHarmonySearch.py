import numpy as np

class DynamicPitchHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.pitch_range = 0.5  # Introduce a pitch adjustment range

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget - len(population)):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            
            pitch_adjustment = np.random.uniform(-self.pitch_range, self.pitch_range, self.dim)
            new_harmony += pitch_adjustment
            
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_fitness = func(new_harmony)
            
            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                population[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]