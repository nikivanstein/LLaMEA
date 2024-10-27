import numpy as np

class HarmonySearchAlgorithm:
    def __init__(self, budget, dim, harmony_memory_size=20, bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth

    def initialize_harmony_memory(self):
        return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

    def improvise_harmony(self, harmony_memory):
        new_harmony = np.copy(harmony_memory[np.random.randint(self.harmony_memory_size)])
        for i in range(self.dim):
            if np.random.rand() < self.bandwidth:
                new_harmony[i] = np.random.uniform(-5.0, 5.0)
        return new_harmony

    def __call__(self, func):
        harmony_memory = self.initialize_harmony_memory()
        harmony_fitness = np.array([func(individual) for individual in harmony_memory])

        for _ in range(self.budget):
            new_harmony = self.improvise_harmony(harmony_memory)
            new_fitness = func(new_harmony)
            worst_idx = np.argmax(harmony_fitness)
            
            if new_fitness < harmony_fitness[worst_idx]:
                harmony_memory[worst_idx] = new_harmony
                harmony_fitness[worst_idx] = new_fitness
        
        return harmony_memory[np.argmin(harmony_fitness)]