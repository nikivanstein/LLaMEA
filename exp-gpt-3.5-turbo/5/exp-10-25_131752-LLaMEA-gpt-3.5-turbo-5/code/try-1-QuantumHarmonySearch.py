import numpy as np

class QuantumHarmonySearch:
    def __init__(self, budget, dim, hmcr=0.7, par=0.5, bw=0.01, qmr=0.5):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.qmr = qmr

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        def quantum_mutation(harmony_memory):
            mutated_harmony = np.copy(harmony_memory)
            for i in range(self.dim):
                if np.random.rand() < self.qmr:
                    phase_shift = np.random.uniform(0, 2*np.pi)
                    mutated_harmony[i] += np.sin(phase_shift) * np.random.uniform(-1, 1)
                    mutated_harmony[i] = np.clip(mutated_harmony[i], -5.0, 5.0)
            return mutated_harmony

        harmony_memory = initialize_harmony_memory()
        for _ in range(self.budget):
            new_harmony = quantum_mutation(harmony_memory)
            if func(new_harmony) < func(harmony_memory[0]):
                harmony_memory[0] = new_harmony
        return harmony_memory[0]