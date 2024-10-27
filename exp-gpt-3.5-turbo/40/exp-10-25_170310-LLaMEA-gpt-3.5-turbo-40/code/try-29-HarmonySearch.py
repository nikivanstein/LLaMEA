import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.bandwidth = 0.01
        self.harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            new_solution = self.harmony_memory[np.random.randint(self.harmony_memory_size)]
            for i in range(self.dim):
                if np.random.rand() < 0.35:
                    new_solution[i] = np.clip(new_solution[i] + np.random.normal(0, self.bandwidth), -5.0, 5.0)
            if func(new_solution) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_solution
            self.harmony_memory = self.harmony_memory[np.argsort([func(x) for x in self.harmony_memory])]
        return self.harmony_memory[0]