import numpy as np

class HarmonySearchOptimizer:
    def __init__(self, budget, dim, harmony_memory_size=20, bandwidth=0.01, pitch_adjust_rate=0.3):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.pitch_adjust_rate = pitch_adjust_rate

    def __call__(self, func):
        harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        best_solution = harmony_memory[np.argmin([func(individual) for individual in harmony_memory])]
        for _ in range(self.budget):
            new_solution = np.clip(best_solution + np.random.uniform(-self.bandwidth, self.bandwidth, self.dim), -5.0, 5.0)
            if func(new_solution) < func(best_solution):
                best_solution = new_solution
            else:
                idx = np.random.randint(self.harmony_memory_size)
                new_solution = np.clip(harmony_memory[idx] + self.pitch_adjust_rate * (best_solution - harmony_memory[idx]) + np.random.uniform(-self.bandwidth, self.bandwidth, self.dim), -5.0, 5.0)
                if func(new_solution) < func(harmony_memory[idx]):
                    harmony_memory[idx] = new_solution

        return best_solution