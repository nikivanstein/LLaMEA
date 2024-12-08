import numpy as np

class EnhancedHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim, harmony_memory_size=10, bandwidth_range=(0.01, 0.1)):
        super().__init__(budget, dim, harmony_memory_size, bandwidth_range[0])
        self.bandwidth_range = bandwidth_range

    def __call__(self, func):
        def improvise(harmony_memory, bandwidth):
            new_solution = np.copy(harmony_memory[np.random.randint(self.harmony_memory_size)])
            for i in range(self.dim):
                if np.random.rand() < bandwidth:
                    new_solution[i] = np.random.uniform(-5.0, 5.0)
            return new_solution

        harmony_memory = self.initialize_harmony_memory()
        for _ in range(self.budget):
            bandwidth = self.bandwidth_range[0] + (_ / self.budget) * (self.bandwidth_range[1] - self.bandwidth_range[0])
            new_solution = improvise(harmony_memory, bandwidth)
            if func(new_solution) < func(harmony_memory[-1]):
                harmony_memory = self.update_harmony_memory(harmony_memory, new_solution)

        return harmony_memory[0]