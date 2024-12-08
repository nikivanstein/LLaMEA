import numpy as np

class HarmonySearchWithMemory:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = int(0.1 * budget)
        self.bandwidth = 0.02  # Increased bandwidth for broader exploration
        self.harmony_memory = np.random.uniform(-5.0, 5.0, (self.memory_size, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            new_solution = np.clip(np.random.normal(np.mean(self.harmony_memory, axis=0), self.bandwidth), -5.0, 5.0)
            if func(new_solution) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_solution
                self.harmony_memory = self.harmony_memory[self.harmony_memory[:,0].argsort()]
        return self.harmony_memory[0]