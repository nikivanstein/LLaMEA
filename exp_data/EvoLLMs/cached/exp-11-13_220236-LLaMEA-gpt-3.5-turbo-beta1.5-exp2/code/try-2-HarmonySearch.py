import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def generate_random_solution():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def improvise(harmony_memory, bandwidth):
            new_harmony = np.copy(harmony_memory[np.random.randint(0, len(harmony_memory))])
            for i in range(self.dim):
                if np.random.rand() < bandwidth:
                    new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
            return new_harmony

        harmony_memory = [generate_random_solution() for _ in range(10)]
        bandwidth = 0.9
        for _ in range(self.budget):
            new_harmony = improvise(harmony_memory, bandwidth)
            if func(new_harmony) < func(harmony_memory[-1]):
                harmony_memory[-1] = new_harmony
                harmony_memory.sort(key=func)
        return harmony_memory[0]