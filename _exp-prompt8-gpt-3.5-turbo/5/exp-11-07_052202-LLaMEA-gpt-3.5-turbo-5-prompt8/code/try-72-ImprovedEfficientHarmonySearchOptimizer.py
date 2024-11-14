import numpy as np

class ImprovedEfficientHarmonySearchOptimizer:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjust_rate=0.3, pitch_bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.pitch_bandwidth = pitch_bandwidth
        self.pitch_threshold = self.pitch_adjust_rate * self.harmony_memory_size

    def __call__(self, func):
        harmony_memory = np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))
        fitness_memory = np.array([func(harmony) for harmony in harmony_memory])

        for i in range(self.budget - self.harmony_memory_size):
            new_harmony = np.zeros((self.dim,))
            for d in range(self.dim):
                if np.random.rand() >= self.pitch_threshold:
                    new_harmony[d] = np.random.uniform(-self.pitch_bandwidth, self.pitch_bandwidth)
                else:
                    new_harmony[d] = harmony_memory[np.random.randint(self.harmony_memory_size), d]
            new_fitness = func(new_harmony)
            min_index = np.argmin(fitness_memory)
            if new_fitness < fitness_memory[min_index]:
                harmony_memory[min_index] = new_harmony
                fitness_memory[min_index] = new_fitness

        best_index = np.argmin(fitness_memory)
        return harmony_memory[best_index]

budget = 1000
dim = 10
optimizer = ImprovedEfficientHarmonySearchOptimizer(budget, dim)
optimized_solution = optimizer(lambda x: np.sum(x ** 2))