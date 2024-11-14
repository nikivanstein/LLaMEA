import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 10
        self.bandwidth = 0.01 * (self.upper_bound - self.lower_bound)
    
    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.harmony_memory_size, self.dim))
        harmony_memory_fitness = np.array([func(x) for x in harmony_memory])
        for _ in range(self.budget - self.harmony_memory_size):
            new_harmony = np.array([self.create_new_harmony(harmony_memory) for _ in range(self.dim)])
            new_fitness = func(new_harmony)
            worst_indices = np.argsort(harmony_memory_fitness)[-self.dim:]
            for worst_index in worst_indices:
                if new_fitness < harmony_memory_fitness[worst_index]:
                    harmony_memory[worst_index] = new_harmony
                    harmony_memory_fitness[worst_index] = new_fitness
        return harmony_memory[np.argmin(harmony_memory_fitness)]
    
    def create_new_harmony(self, harmony_memory):
        new_harmony = np.copy(harmony_memory[np.random.randint(self.harmony_memory_size)])
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                new_harmony[i] += np.random.uniform(-self.bandwidth, self.bandwidth)
                new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
        return new_harmony