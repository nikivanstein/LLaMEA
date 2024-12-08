import numpy as np

class EnhancedQuantumInspiredHarmonySearch:
    def __init__(self, budget, dim, hmcr=0.95, par=0.45, bw=0.01, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.f = f
        self.cr = cr

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        def update_harmony_memory(harmony_memory, new_solution):
            min_idx = np.argmin(func(harmony_memory))
            if func(new_solution) < func(harmony_memory[min_idx]):
                harmony_memory[min_idx] = new_solution

        def adjust_parameter(value, lower_bound, upper_bound):
            if value < lower_bound:
                return lower_bound
            elif value > upper_bound:
                return upper_bound
            else:
                return value

        harmony_memory = initialize_harmony_memory()

        for _ in range(self.budget):
            new_solution = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[d] = harmony_memory[np.random.randint(self.budget)][d]
                else:
                    r1, r2, r3 = np.random.choice(self.budget, 3, replace=False)
                    mutant_vector = harmony_memory[r1] + self.f * (harmony_memory[r2] - harmony_memory[r3])
                    crossover_mask = np.random.rand(self.dim) < self.cr
                    new_solution = np.where(crossover_mask, mutant_vector, harmony_memory[np.random.randint(self.budget)])

            update_harmony_memory(harmony_memory, new_solution)

        return harmony_memory[np.argmin(func(harmony_memory))]