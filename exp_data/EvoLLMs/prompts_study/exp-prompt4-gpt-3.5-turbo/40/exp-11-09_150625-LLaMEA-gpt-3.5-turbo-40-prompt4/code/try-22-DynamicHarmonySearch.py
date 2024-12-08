import numpy as np

class DynamicHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.bandwidth_min = 0.01
        self.bandwidth_max = 0.1

    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        harmony_memory_fit = np.apply_along_axis(func, 1, harmony_memory)
        
        for _ in range(self.budget - self.harmony_memory_size):
            new_harmony = np.zeros((1, self.dim))
            current_bandwidth = self.bandwidth_max - (_ / self.budget) * (self.bandwidth_max - self.bandwidth_min)
            for d in range(self.dim):
                if np.random.rand() < 0.7:
                    new_harmony[0, d] = harmony_memory[np.random.randint(self.harmony_memory_size), d]
                else:
                    new_harmony[0, d] = np.random.uniform(self.lower_bound, self.upper_bound)
                    if np.random.rand() < 0.5:
                        new_harmony[0, d] += np.random.uniform(-current_bandwidth, current_bandwidth)
            new_harmony_fit = func(new_harmony)
            if new_harmony_fit < harmony_memory_fit.max():
                replace_idx = np.argmax(harmony_memory_fit)
                harmony_memory[replace_idx] = new_harmony
                harmony_memory_fit[replace_idx] = new_harmony_fit
        
        best_idx = np.argmin(harmony_memory_fit)
        best_solution = harmony_memory[best_idx]
        best_fitness = harmony_memory_fit[best_idx]
        
        return best_solution, best_fitness