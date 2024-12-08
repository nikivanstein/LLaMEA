import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.bandwidth = 0.5

    def __call__(self, func):
        harmony_memory_size = 20
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (harmony_memory_size, self.dim))
        best_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = np.inf

        for _ in range(self.budget):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            for i in range(self.dim):
                if np.random.rand() < 0.7:
                    new_harmony[i] = np.random.choice(harmony_memory[:, i])
                new_harmony[i] += np.random.uniform(-self.bandwidth, self.bandwidth)

            new_fitness = func(new_harmony)
            if new_fitness < best_fitness:
                best_harmony = new_harmony
                best_fitness = new_fitness

            worst_idx = np.argmax(harmony_memory, axis=0)
            if new_fitness < func(harmony_memory[worst_idx]):
                harmony_memory[worst_idx] = new_harmony

        return best_harmony