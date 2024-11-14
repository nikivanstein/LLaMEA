import numpy as np

class ImprovedHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.bandwidth_percent = 0.01
        
    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.harmony_memory_size, self.dim))
        harmony_memory_fitness = np.array([func(x) for x in harmony_memory])
        for _ in range(self.budget - self.harmony_memory_size):
            new_harmony = np.array([self.create_new_harmony(harmony_memory) for _ in range(self.dim)])
            new_fitness = func(new_harmony)
            worst_index = np.argmax(harmony_memory_fitness)
            if new_fitness < harmony_memory_fitness[worst_index]:
                harmony_memory[worst_index] = new_harmony
                harmony_memory_fitness[worst_index] = new_fitness
            self.adjust_bandwidth(new_fitness)
        return harmony_memory[np.argmin(harmony_memory_fitness)]
    
    def adjust_bandwidth(self, new_fitness):
        if np.random.rand() < 0.05:
            improvement_rate = (np.min(new_fitness) - np.max(new_fitness)) / self.budget
            self.bandwidth *= 1 + improvement_rate
            self.bandwidth = min(self.bandwidth, 0.1 * (self.upper_bound - self.lower_bound))