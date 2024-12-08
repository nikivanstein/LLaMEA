import numpy as np

class ImprovedHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.bandwidth = 0.5

    def __call__(self, func):
        harmony_memory_size = 20
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (harmony_memory_size, self.dim))
        best_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = np.inf

        for _ in range(self.budget):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.bandwidth:
                    new_harmony[i] = np.random.choice(harmony_memory[:, i])

            new_fitness = func(new_harmony)
            if new_fitness < best_fitness:
                best_harmony = new_harmony
                best_fitness = new_fitness

            worst_idx = np.argmax(harmony_memory, axis=0)
            if new_fitness < func(harmony_memory[worst_idx]):
                harmony_memory[worst_idx] = new_harmony

            self.bandwidth = max(0.3, self.bandwidth * 0.99)  # Adaptive bandwidth update

        return best_harmony