import numpy as np

class EnhancedHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.pitch_adjustment_rate = 0.1

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget - len(population)):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            new_harmony = self._adjust_pitch(new_harmony)
            new_fitness = func(new_harmony)
            
            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                population[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]

    def _adjust_pitch(self, harmony):
        for i in range(len(harmony)):
            if np.random.rand() < self.pitch_adjustment_rate:
                harmony[i] += np.random.uniform(-0.5, 0.5) * (self.upper_bound - self.lower_bound)
                harmony[i] = min(max(harmony[i], self.lower_bound), self.upper_bound)
        return harmony