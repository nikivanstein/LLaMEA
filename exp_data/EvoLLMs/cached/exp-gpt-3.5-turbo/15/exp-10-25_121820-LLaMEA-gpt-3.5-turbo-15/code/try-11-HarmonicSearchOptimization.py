import numpy as np

class HarmonicSearchOptimization:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjustment_rate=0.1, bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjustment_rate = pitch_adjustment_rate
        self.bandwidth = bandwidth

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

        harmony_memory = initialize_harmony_memory()
        harmony_fitness = np.array([func(harmony) for harmony in harmony_memory])
        for _ in range(self.budget - self.harmony_memory_size):
            new_harmony = []
            for i in range(self.dim):
                if np.random.rand() < self.pitch_adjustment_rate:
                    new_value = np.random.uniform(-self.bandwidth, self.bandwidth)
                else:
                    new_value = harmony_memory[np.random.choice(range(self.harmony_memory_size))][i]
                new_harmony.append(new_value)
            new_fitness = func(new_harmony)
            worst_idx = np.argmax(harmony_fitness)
            if new_fitness < harmony_fitness[worst_idx]:
                harmony_memory[worst_idx] = new_harmony
                harmony_fitness[worst_idx] = new_fitness

        best_idx = np.argmin(harmony_fitness)
        return harmony_memory[best_idx]