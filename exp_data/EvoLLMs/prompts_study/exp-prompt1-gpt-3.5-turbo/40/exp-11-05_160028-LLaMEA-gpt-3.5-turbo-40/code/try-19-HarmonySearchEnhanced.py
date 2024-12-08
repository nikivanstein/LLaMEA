import numpy as np

class HarmonySearchEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pitch_adjustment_rate = 0.5

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
                adjust_amount = self.pitch_adjustment_rate * np.random.uniform(-1, 1, self.dim)
                new_harmony = np.clip(new_harmony + adjust_amount, self.lower_bound, self.upper_bound)
                new_fitness = func(new_harmony)

                if new_fitness < fitness[worst_idx]:
                    population[worst_idx] = new_harmony
                    fitness[worst_idx] = new_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx]