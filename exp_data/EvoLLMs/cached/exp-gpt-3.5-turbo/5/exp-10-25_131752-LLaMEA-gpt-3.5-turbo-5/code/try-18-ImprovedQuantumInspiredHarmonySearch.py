import numpy as np

class ImprovedQuantumInspiredHarmonySearch:
    def __init__(self, budget, dim, hmcr=0.7, par=0.5, bw=0.01, phase_factor=0.5, adapt_rate=0.05):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.phase_factor = phase_factor
        self.adapt_rate = adapt_rate

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        def apply_quantum_phase(harmony_memory, phase_factor):
            phases = np.random.uniform(0, 2 * np.pi, size=self.dim)
            return np.multiply(harmony_memory, np.exp(1j * phase_factor * phases))

        def improvise_new_harmony(harmony_memory, phase_factor):
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
            return apply_quantum_phase(new_harmony, phase_factor)

        harmony_memory = initialize_harmony_memory()
        current_phase_factor = self.phase_factor
        for _ in range(self.budget):
            new_harmony = improvise_new_harmony(harmony_memory, current_phase_factor)
            if func(new_harmony) < func(harmony_memory[0]):
                harmony_memory[0] = new_harmony
            current_phase_factor = max(0, current_phase_factor + self.adapt_rate * (1 - _ / self.budget))
        return harmony_memory[0]