import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.bandwidth = 0.01 * (self.upper_bound - self.lower_bound)
        self.bandwidth_min = 0.01 * (self.upper_bound - self.lower_bound)
        self.bandwidth_max = 0.1 * (self.upper_bound - self.lower_bound)
        self.convergence_threshold = 0.05
        self.convergence_rate = 0.0
    
    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.dim,))
        harmony_memory_fitness = func(harmony_memory)
        for _ in range(self.budget - 1):
            new_harmony = self.create_new_harmony(harmony_memory)
            new_fitness = func(new_harmony)
            if new_fitness < harmony_memory_fitness:
                harmony_memory = new_harmony
                harmony_memory_fitness = new_fitness
            self.update_bandwidth(new_fitness, harmony_memory_fitness)
        return harmony_memory
    
    def create_new_harmony(self, harmony_memory):
        new_harmony = np.copy(harmony_memory)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                new_harmony[i] += np.random.uniform(-self.bandwidth, self.bandwidth)
                new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
        return new_harmony
    
    def update_bandwidth(self, new_fitness, harmony_memory_fitness):
        current_convergence_rate = abs(new_fitness - harmony_memory_fitness) / harmony_memory_fitness
        self.convergence_rate = 0.8 * self.convergence_rate + 0.2 * current_convergence_rate
        if self.convergence_rate < self.convergence_threshold:
            self.bandwidth = min(self.bandwidth_max, self.bandwidth * 1.1)
        else:
            self.bandwidth = max(self.bandwidth_min, self.bandwidth * 0.9)