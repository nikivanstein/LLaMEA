import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 5
        self.pitch_adjustment_rate = 0.5
        self.bandwidth = 0.01

    def _initialize_population(self):
        return np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))

    def _local_search(self, harmony):
        new_harmony = np.copy(harmony)
        for i in range(self.dim):
            new_harmony[i] += np.random.uniform(-self.bandwidth, self.bandwidth)
            new_harmony[i] = np.clip(new_harmony[i], -5.0, 5.0)
        return new_harmony

    def __call__(self, func):
        population = self._initialize_population()

        for _ in range(self.budget // self.harmony_memory_size):
            new_harmonies = [self._local_search(harmony) for harmony in population]
            combined_population = np.vstack((population, new_harmonies))
            scores = np.array([func(individual) for individual in combined_population])
            sorted_indices = np.argsort(scores)
            population = combined_population[sorted_indices[:self.harmony_memory_size]]

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution