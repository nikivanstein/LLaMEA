import numpy as np

class EnhancedQuantumInspiredHarmonySearch(QuantumInspiredHarmonySearch):
    def __init__(self, budget, dim, hmcr=0.95, par=0.45, bw=0.01, par_min=0.01, par_max=0.5, par_decay=0.95):
        super().__init__(budget, dim, hmcr, par, bw)
        self.par_min = par_min
        self.par_max = par_max
        self.par_decay = par_decay

    def __call__(self, func):
        def adjust_pitch_adjustment_rate():
            return max(self.par_min, self.par * self.par_decay)

        harmony_memory = self.initialize_harmony_memory()

        for _ in range(self.budget):
            new_solution = np.zeros(self.dim)
            current_par = self.adjust_parameter(self.par, self.par_min, self.par_max)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[d] = harmony_memory[np.random.randint(self.budget)][d]
                else:
                    new_solution[d] = np.clip(harmony_memory[np.random.randint(self.budget)][d] + np.random.normal(0, 1) * current_par, -5.0, 5.0)
            self.update_harmony_memory(harmony_memory, new_solution)
            self.par = adjust_pitch_adjustment_rate()

        return harmony_memory[np.argmin(func(harmony_memory))]