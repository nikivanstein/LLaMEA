import numpy as np

class DynamicHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pitch_adjust_rate = 0.5  # New parameter for pitch adjustment

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        pitch_bandwidth = (self.upper_bound - self.lower_bound) * self.pitch_adjust_rate

        for _ in range(self.budget - len(population)):
            new_harmony = np.array([np.random.uniform(max(self.lower_bound, h - pitch_bandwidth), min(self.upper_bound, h + pitch_bandwidth)) for h in population[np.random.choice(len(population))]])
            new_fitness = func(new_harmony)
            
            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                population[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]