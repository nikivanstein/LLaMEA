import numpy as np

class ImprovedQuantumInspiredHarmonySearch(QuantumInspiredHarmonySearch):
    def __init__(self, budget, dim, hmcr=0.95, par=0.45, bw=0.01, pitch_adj_rate=0.03):
        super().__init__(budget, dim, hmcr, par, bw)
        self.pitch_adj_rate = pitch_adj_rate

    def __call__(self, func):
        def adjust_pitch_adjustment_rate():
            return np.clip(self.pitch_adj_rate * np.exp(-0.1 * np.random.rand()), 0.001, 0.1)

        harmony_memory = self.initialize_harmony_memory()
        pitch_adj_rate = self.pitch_adj_rate

        for _ in range(self.budget):
            new_solution = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[d] = harmony_memory[np.random.randint(self.budget)][d]
                else:
                    new_solution[d] = np.clip(harmony_memory[np.random.randint(self.budget)][d] + np.random.normal(0, 1) * pitch_adj_rate, -5.0, 5.0)
            self.update_harmony_memory(harmony_memory, new_solution)
            pitch_adj_rate = adjust_pitch_adjustment_rate()

        return harmony_memory[np.argmin(func(harmony_memory))]