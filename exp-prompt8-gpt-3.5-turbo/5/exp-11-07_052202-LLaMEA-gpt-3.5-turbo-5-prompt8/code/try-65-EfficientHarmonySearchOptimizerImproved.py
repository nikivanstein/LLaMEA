import numpy as np

class EfficientHarmonySearchOptimizerImproved:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjust_rate=0.3, pitch_bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.pitch_bandwidth = pitch_bandwidth
        self.pitch_threshold = self.pitch_adjust_rate * self.harmony_memory_size
        self.new_harmony = np.zeros((self.dim,))
        self.new_fitness = 0.0

    def __call__(self, func):
        self.harmony_memory = np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))
        self.fitness_memory = np.apply_along_axis(func, 1, self.harmony_memory)

        for _ in range(self.budget - self.harmony_memory_size):
            for d in range(self.dim):
                if np.random.rand() >= self.pitch_threshold:
                    self.new_harmony[d] = np.random.uniform(-self.pitch_bandwidth, self.pitch_bandwidth)
                else:
                    idx = np.random.randint(self.harmony_memory_size)
                    self.new_harmony[d] = self.harmony_memory[idx, d]
            self.new_fitness = func(self.new_harmony)
            min_idx = np.argmin(self.fitness_memory)
            if self.new_fitness < self.fitness_memory[min_idx]:
                self.harmony_memory[min_idx] = np.copy(self.new_harmony)
                self.fitness_memory[min_idx] = self.new_fitness

        best_idx = np.argmin(self.fitness_memory)
        return self.harmony_memory[best_idx]