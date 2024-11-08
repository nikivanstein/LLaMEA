import numpy as np

class FastHarmonySearchOptimizer:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjust_rate=0.3, pitch_bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.pitch_bandwidth = pitch_bandwidth
        self.pitch_threshold = self.pitch_adjust_rate * self.harmony_memory_size

    def __call__(self, func):
        self.harmony_memory = np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))
        self.fitness_memory = np.array([func(harmony) for harmony in self.harmony_memory])

        for _ in range(self.budget - self.harmony_memory_size):
            mask = np.random.rand(self.dim) >= self.pitch_threshold
            new_harmony = np.where(mask, np.random.uniform(-self.pitch_bandwidth, self.pitch_bandwidth, self.dim), np.take(self.harmony_memory, np.random.randint(self.harmony_memory_size), axis=0))
            new_fitness = func(new_harmony)
            min_index = np.argmin(self.fitness_memory)
            mask_fitness = new_fitness < self.fitness_memory[min_index]
            self.harmony_memory[min_index] = np.where(mask_fitness, new_harmony, self.harmony_memory[min_index])
            self.fitness_memory[min_index] = np.where(mask_fitness, new_fitness, self.fitness_memory[min_index])

        best_index = np.argmin(self.fitness_memory)
        return self.harmony_memory[best_index]

# Usage:
budget = 1000
dim = 10
optimizer = FastHarmonySearchOptimizer(budget, dim)
optimized_solution = optimizer(lambda x: np.sum(x ** 2))