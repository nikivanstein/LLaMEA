import numpy as np

class EnhancedQuantumInspiredHarmonySearch(QuantumInspiredHarmonySearch):
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
                if np.random.rand() < adjust_parameter(self.hmcr, 0, 1):
                    new_solution[d] = harmony_memory[np.random.randint(self.budget)][d]
                else:
                    new_solution[d] = np.clip(harmony_memory[np.random.randint(self.budget)][d] + np.random.normal(0, 1) * self.bw, -5.0, 5.0)
            update_harmony_memory(harmony_memory, new_solution)
            self.hmcr = adjust_parameter(self.hmcr - 0.001, 0, 1)  # Adaptive parameter tuning

        return harmony_memory[np.argmin(func(harmony_memory))]