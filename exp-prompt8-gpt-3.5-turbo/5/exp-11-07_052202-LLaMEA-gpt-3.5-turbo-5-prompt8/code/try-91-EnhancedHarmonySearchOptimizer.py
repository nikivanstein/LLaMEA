import numpy as np

class EnhancedHarmonySearchOptimizer:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjust_rate=0.3, pitch_bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.pitch_bandwidth = pitch_bandwidth
        self.pitch_threshold = self.pitch_adjust_rate * self.harmony_memory_size
        self.harmony_memory = np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))
        self.fitness_memory = np.array([func(harmony) for harmony in self.harmony_memory])
    
    def __call__(self, func):
        for _ in range(self.budget - self.harmony_memory_size):
            pitch_mask = np.random.rand(self.dim) >= self.pitch_threshold
            new_harmony = self.harmony_memory[np.random.randint(self.harmony_memory_size, size=(self.dim)), np.arange(self.dim)] * pitch_mask[:, np.newaxis] + np.random.uniform(-self.pitch_bandwidth, self.pitch_bandwidth, (self.dim,)) * (~pitch_mask)[:, np.newaxis]
            new_fitness = func(new_harmony)
            min_index = np.argmin(self.fitness_memory)
            update_mask = new_fitness < self.fitness_memory[min_index]
            self.harmony_memory[min_index] = np.where(update_mask, new_harmony, self.harmony_memory[min_index])
            self.fitness_memory[min_index] = np.where(update_mask, new_fitness, self.fitness_memory[min_index])
        
        return self.harmony_memory[np.argmin(self.fitness_memory)]

budget = 1000
dim = 10
optimizer = EnhancedHarmonySearchOptimizer(budget, dim)
optimized_solution = optimizer(lambda x: np.sum(x ** 2))