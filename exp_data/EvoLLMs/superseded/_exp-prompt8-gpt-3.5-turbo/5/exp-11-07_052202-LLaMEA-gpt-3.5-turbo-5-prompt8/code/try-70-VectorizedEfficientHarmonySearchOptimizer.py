import numpy as np

class VectorizedEfficientHarmonySearchOptimizer:
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
        self.fitness_memory = np.array([func(harmony) for harmony in self.harmony_memory])

        for _ in range(self.budget - self.harmony_memory_size):
            pitch_mask = np.random.rand(self.dim) >= self.pitch_threshold
            pitch_indices = np.where(pitch_mask)
            self.new_harmony[pitch_indices] = np.random.uniform(-self.pitch_bandwidth, self.pitch_bandwidth, size=len(pitch_indices[0]))
            harmony_indices = np.random.randint(0, self.harmony_memory_size, size=self.dim - len(pitch_indices[0]))
            self.new_harmony[harm_indices] = self.harmony_memory[harm_indices]
            self.new_fitness = func(self.new_harmony)
            min_index = np.argmin(self.fitness_memory)
            if self.new_fitness < self.fitness_memory[min_index]:
                self.harmony_memory[min_index] = np.copy(self.new_harmony)
                self.fitness_memory[min_index] = self.new_fitness

        best_index = np.argmin(self.fitness_memory)
        return self.harmony_memory[best_index]

budget = 1000
dim = 10
optimizer = VectorizedEfficientHarmonySearchOptimizer(budget, dim)
optimized_solution = optimizer(lambda x: np.sum(x ** 2))