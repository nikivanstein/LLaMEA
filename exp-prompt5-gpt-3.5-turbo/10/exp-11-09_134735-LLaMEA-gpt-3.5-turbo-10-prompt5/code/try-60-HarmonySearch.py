import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hmcr = 0.7  # Harmony Memory Consideration Rate
        self.par = 0.3  # Pitch Adjustment Rate

    def __call__(self, func):
        harmony_memory_size = 20
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (harmony_memory_size, self.dim))
        best_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = np.inf

        for _ in range(self.budget):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[i] = np.random.choice(harmony_memory[:, i])
                if np.random.rand() < self.par:
                    new_harmony[i] += np.random.uniform(-1, 1)

            new_fitness = func(new_harmony)
            if new_fitness < best_fitness:
                best_harmony = new_harmony
                best_fitness = new_fitness

            worst_idx = np.argmax(harmony_memory, axis=0)
            if new_fitness < func(harmony_memory[worst_idx]):
                harmony_memory[worst_idx] = new_harmony

        return best_harmony