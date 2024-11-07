import numpy as np

class EnhancedQuantumInspiredHarmonySearch(QuantumInspiredHarmonySearch):
    def __init__(self, budget, dim, hmcr=0.95, par=0.45, bw=0.01, opp_prob=0.1):
        super().__init__(budget, dim, hmcr, par, bw)
        self.opp_prob = opp_prob

    def __call__(self, func):
        def opposition_based_learning(solution):
            return np.where(np.random.rand(self.dim) < self.opp_prob, -solution, solution)

        harmony_memory = self.initialize_harmony_memory()

        for _ in range(self.budget):
            new_solution = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[d] = harmony_memory[np.random.randint(self.budget)][d]
                else:
                    new_solution[d] = np.clip(harmony_memory[np.random.randint(self.budget)][d] + np.random.normal(0, 1) * self.bw, -5.0, 5.0)
                new_solution = opposition_based_learning(new_solution)
            self.update_harmony_memory(harmony_memory, new_solution)

        return harmony_memory[np.argmin(func(harmony_memory))]