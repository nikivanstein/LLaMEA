import numpy as np

class EnhancedQuantumInspiredHarmonySearch(QuantumInspiredHarmonySearch):
    def __init__(self, budget, dim, hmcr=0.95, par=0.45, bw=0.01, bw_lower=0.01, bw_upper=0.1, bw_decay=0.95):
        super().__init__(budget, dim, hmcr, par, bw)
        self.bw_lower = bw_lower
        self.bw_upper = bw_upper
        self.bw_decay = bw_decay

    def __call__(self, func):
        def adjust_bandwidth():
            return max(self.bw_lower, self.bw * self.bw_decay)

        def update_bandwidth():
            self.bw = adjust_parameter(self.bw * self.bw_decay, self.bw_lower, self.bw_upper)

        def adjust_parameter(value, lower_bound, upper_bound):
            return max(lower_bound, min(value, upper_bound))

        harmony_memory = initialize_harmony_memory()

        for _ in range(self.budget):
            new_solution = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[d] = harmony_memory[np.random.randint(self.budget)][d]
                else:
                    new_solution[d] = np.clip(harmony_memory[np.random.randint(self.budget)][d] + np.random.normal(0, 1) * self.bw, -5.0, 5.0)
            update_harmony_memory(harmony_memory, new_solution)
            update_bandwidth()

        return harmony_memory[np.argmin(func(harmony_memory))]