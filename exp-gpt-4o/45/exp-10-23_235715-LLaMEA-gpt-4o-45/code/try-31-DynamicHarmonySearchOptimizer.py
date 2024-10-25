import numpy as np

class DynamicHarmonySearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = min(30, self.budget // (3 * dim))
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluations = 0
        self.harmony_memory_consideration_rate = 0.95
        self.pitch_adjustment_rate = 0.1
        self.adjustment_bandwidth = 0.01

    def adaptive_parameters(self):
        self.harmony_memory_consideration_rate = 0.8 + 0.15 * np.random.rand()
        self.pitch_adjustment_rate = 0.05 + 0.1 * np.random.rand()
        self.adjustment_bandwidth = 0.005 + 0.015 * np.random.rand()

    def harmony_search(self, func):
        for _ in range(self.budget // (2 * self.harmony_memory_size)):
            if self.evaluations >= self.budget:
                break
            self.adaptive_parameters()
            for i in range(self.harmony_memory_size):
                if self.evaluations >= self.budget:
                    return
                new_harmony = np.zeros(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.harmony_memory_consideration_rate:
                        new_harmony[j] = self.harmony_memory[np.random.randint(self.harmony_memory_size)][j]
                        if np.random.rand() < self.pitch_adjustment_rate:
                            new_harmony[j] += self.adjustment_bandwidth * (np.random.rand() - 0.5)
                    else:
                        new_harmony[j] = np.random.uniform(self.lower_bound, self.upper_bound)
                new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
                new_fitness = func(new_harmony)
                self.evaluations += 1
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_harmony
                worst_index = np.argmax([func(harmony) for harmony in self.harmony_memory])
                if new_fitness < func(self.harmony_memory[worst_index]):
                    self.harmony_memory[worst_index] = new_harmony

    def __call__(self, func):
        self.harmony_search(func)
        return self.best_solution