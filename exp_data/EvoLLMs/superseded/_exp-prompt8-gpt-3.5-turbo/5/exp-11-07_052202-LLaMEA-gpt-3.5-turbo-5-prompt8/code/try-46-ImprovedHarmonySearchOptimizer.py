import numpy as np

class ImprovedHarmonySearchOptimizer:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjust_rate=0.3, pitch_bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.pitch_bandwidth = pitch_bandwidth
        self.pitch_threshold = self.pitch_adjust_rate * self.harmony_memory_size
        self.harmony_memory = np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))
        self.fitness_memory = np.empty(self.harmony_memory_size)
        for i in range(self.harmony_memory_size):
            self.fitness_memory[i] = func(self.harmony_memory[i])

    def __call__(self, func):
        for _ in range(self.budget - self.harmony_memory_size):
            new_harmony = np.empty(self.dim)
            for d in range(self.dim):
                if np.random.rand() >= self.pitch_threshold:
                    new_harmony[d] = np.random.uniform(-self.pitch_bandwidth, self.pitch_bandwidth)
                else:
                    new_harmony[d] = self.harmony_memory[np.random.randint(self.harmony_memory_size), d]
            new_fitness = func(new_harmony)
            min_index = np.argmin(self.fitness_memory)
            if new_fitness < self.fitness_memory[min_index]:
                self.harmony_memory[min_index] = new_harmony
                self.fitness_memory[min_index] = new_fitness

        best_index = np.argmin(self.fitness_memory)
        return self.harmony_memory[best_index]

# Usage:
budget = 1000
dim = 10
optimizer = ImprovedHarmonySearchOptimizer(budget, dim)
optimized_solution = optimizer(lambda x: np.sum(x ** 2))