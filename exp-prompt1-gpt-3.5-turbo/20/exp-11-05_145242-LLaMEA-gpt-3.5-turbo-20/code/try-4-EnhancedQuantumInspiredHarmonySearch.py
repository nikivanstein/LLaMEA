import numpy as np

class EnhancedQuantumInspiredHarmonySearch(QuantumInspiredHarmonySearch):
    def __init__(self, budget, dim, hmcr=0.95, par=0.45, bw=0.01, bw_min=0.001, bw_max=0.1, bw_decay=0.99):
        super().__init__(budget, dim, hmcr, par, bw)
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.bw_decay = bw_decay

    def __call__(self, func):
        def adjust_bandwidth():
            return max(self.bw_min, min(self.bw * self.bw_decay, self.bw_max))

        def update_harmony_memory(harmony_memory, new_solution, bandwidth):
            min_idx = np.argmin(func(harmony_memory))
            if func(new_solution) < func(harmony_memory[min_idx]):
                harmony_memory[min_idx] = new_solution
                return True
            return False

        harmony_memory = self.initialize_harmony_memory()

        for _ in range(self.budget):
            new_solution = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[d] = harmony_memory[np.random.randint(self.budget)][d]
                else:
                    new_solution[d] = np.clip(harmony_memory[np.random.randint(self.budget)][d] + np.random.normal(0, 1) * self.bw, -5.0, 5.0)
            if update_harmony_memory(harmony_memory, new_solution, self.bw):
                self.bw = adjust_bandwidth()

        return harmony_memory[np.argmin(func(harmony_memory))]