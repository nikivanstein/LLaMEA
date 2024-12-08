import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, bandwidth=0.01, pitch_adjust_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.pitch_adjust_rate = pitch_adjust_rate
        self.harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            new_solution = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.pitch_adjust_rate:
                    idx = np.random.randint(self.harmony_memory_size)
                    new_solution[i] = self.harmony_memory[idx, i]
                else:
                    dynamic_bandwidth = self.bandwidth * (1 - _ / self.budget)  # Dynamic bandwidth adjustment
                    if np.random.rand() < 0.9:  # Introduce Levy flight mutation with 90% probability
                        step = np.random.standard_cauchy() / np.sqrt(np.abs(np.random.randn()))
                        new_solution[i] = self.harmony_memory[-1, i] + step
                    else:
                        new_solution[i] = np.random.uniform(-dynamic_bandwidth, dynamic_bandwidth) + np.min(self.harmony_memory[:, i])
                    new_solution[i] = max(-5.0, min(5.0, new_solution[i]))

            if func(new_solution) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_solution
                self.harmony_memory = self.harmony_memory[np.argsort([func(sol) for sol in self.harmony_memory])]

        return self.harmony_memory[0]