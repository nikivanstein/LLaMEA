import numpy as np

class DynamicHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim, harmony_memory_size=10, initial_bandwidth=0.01, bandwidth_decay=0.95):
        super().__init__(budget, dim, harmony_memory_size, initial_bandwidth)
        self.bandwidth_decay = bandwidth_decay

    def __call__(self, func):
        def improvise(harmony_memory, bandwidth):
            new_solution = np.copy(harmony_memory[np.random.randint(self.harmony_memory_size)])
            for i in range(self.dim):
                if np.random.rand() < bandwidth:
                    new_solution[i] = np.random.uniform(-5.0, 5.0)
            return new_solution

        harmony_memory = self.initialize_harmony_memory()
        bandwidth = self.bandwidth
        for _ in range(self.budget):
            new_solution = improvise(harmony_memory, bandwidth)
            if func(new_solution) < func(harmony_memory[-1]):
                harmony_memory = self.update_harmony_memory(harmony_memory, new_solution)
                bandwidth *= self.bandwidth_decay

        return harmony_memory[0]