import numpy as np

class AdaptiveQuantumInspiredHarmonySearch(QuantumInspiredHarmonySearch):
    def __init__(self, budget, dim, hmcr=0.95, par=0.45, bw=0.01, step_size=1.0):
        super().__init__(budget, dim, hmcr, par, bw)
        self.step_size = step_size

    def update_harmony_memory(self, harmony_memory, new_solution):
        min_idx = np.argmin(func(harmony_memory))
        if func(new_solution) < func(harmony_memory[min_idx]):
            harmony_memory[min_idx] = new_solution
            self.step_size *= 0.9  # Adjust step size based on successful update
        else:
            self.step_size *= 1.1  # Increase step size for unsuccessful updates

    def __call__(self, func):
        harmony_memory = initialize_harmony_memory()

        for _ in range(self.budget):
            new_solution = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[d] = harmony_memory[np.random.randint(self.budget)][d]
                else:
                    new_solution[d] = np.clip(harmony_memory[np.random.randint(self.budget)][d] + np.random.normal(0, self.step_size) * self.bw, -5.0, 5.0)
            self.update_harmony_memory(harmony_memory, new_solution)

        return harmony_memory[np.argmin(func(harmony_memory))]