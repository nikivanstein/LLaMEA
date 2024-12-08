import numpy as np

class QuantumInspiredHarmonySearchDynamicBw:
    def __init__(self, budget, dim, hmcr=0.7, par=0.5, bw=0.01, phase_factor=0.5, bw_lower=0.001, bw_upper=0.1, bw_decay=0.95):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.phase_factor = phase_factor
        self.bw_lower = bw_lower
        self.bw_upper = bw_upper
        self.bw_decay = bw_decay

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        def apply_quantum_phase(harmony_memory):
            phase = np.random.uniform(0, 2 * np.pi)
            return np.multiply(harmony_memory, np.exp(1j * self.phase_factor * phase))

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
            return apply_quantum_phase(new_harmony)

        harmony_memory = initialize_harmony_memory()
        for _ in range(self.budget):
            new_harmony = improvise_new_harmony(harmony_memory)
            if func(new_harmony) < func(harmony_memory[0]):
                harmony_memory[0] = new_harmony
            self.bw = max(self.bw_lower, self.bw * self.bw_decay) if func(new_harmony) < func(harmony_memory[0]) else min(self.bw_upper, self.bw / self.bw_decay)
        return harmony_memory[0]