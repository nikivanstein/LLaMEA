import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, hmcr=0.7, par=0.5, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        def improvise_new_harmony(harmony_memory):
            new_harmony = np.copy(harmony_memory)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    if np.random.rand() < self.par:
                        new_harmony[i] = np.random.uniform(-5.0, 5.0)
                    else:
                        j = np.random.randint(self.budget)
                        new_harmony[i] = harmony_memory[j, i]
                else:
                    new_harmony[i] += self.bw * np.random.randn()
                    new_harmony[i] = np.clip(new_harmony[i], -5.0, 5.0)
            return new_harmony

        harmony_memory = initialize_harmony_memory()
        for _ in range(self.budget):
            new_harmony = improvise_new_harmony(harmony_memory)
            if func(new_harmony) < func(harmony_memory[0]):
                harmony_memory[0] = new_harmony
        return harmony_memory[0]