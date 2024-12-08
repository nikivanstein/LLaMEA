import numpy as np

class ImprovedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 10
        self.bandwidth = 0.01
        self.elite_memory_size = 3
        self.elite_memory = []

    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        harmony_memory_fit = np.apply_along_axis(func, 1, harmony_memory)
        
        for _ in range(self.budget - self.harmony_memory_size):
            new_harmony = np.zeros((1, self.dim))
            for d in range(self.dim):
                if np.random.rand() < 0.7:
                    new_harmony[0, d] = harmony_memory[np.random.randint(self.harmony_memory_size), d]
                else:
                    new_harmony[0, d] = np.random.uniform(self.lower_bound, self.upper_bound)
                    if np.random.rand() < 0.5:
                        new_harmony[0, d] += np.random.uniform(-self.bandwidth, self.bandwidth)
            new_harmony_fit = func(new_harmony)
            if len(self.elite_memory) < self.elite_memory_size or new_harmony_fit < self.elite_memory_fit.min():
                if len(self.elite_memory) == self.elite_memory_size:
                    replace_idx = np.argmin(self.elite_memory_fit)
                    self.elite_memory[replace_idx] = new_harmony
                    self.elite_memory_fit[replace_idx] = new_harmony_fit
                else:
                    self.elite_memory.append(new_harmony)
                    self.elite_memory_fit.append(new_harmony_fit)

            if new_harmony_fit < harmony_memory_fit.max():
                replace_idx = np.argmax(harmony_memory_fit)
                harmony_memory[replace_idx] = new_harmony
                harmony_memory_fit[replace_idx] = new_harmony_fit
        
        best_idx = np.argmin(self.elite_memory_fit)
        best_solution = self.elite_memory[best_idx]
        best_fitness = self.elite_memory_fit[best_idx]
        
        return best_solution, best_fitness