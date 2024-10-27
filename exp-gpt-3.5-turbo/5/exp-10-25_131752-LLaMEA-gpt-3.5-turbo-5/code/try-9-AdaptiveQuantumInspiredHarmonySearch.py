import numpy as np

class AdaptiveQuantumInspiredHarmonySearch(QuantumInspiredHarmonySearch):
    def __init__(self, budget, dim, hmcr=0.7, par=0.5, bw=0.01, phase_factor=0.5, phase_update_rate=0.01):
        super().__init__(budget, dim, hmcr, par, bw, phase_factor)
        self.phase_update_rate = phase_update_rate

    def __call__(self, func):
        def apply_quantum_phase(harmony_memory):
            phase = np.random.uniform(0, 2 * np.pi)
            phase_update = np.random.uniform(-self.phase_update_rate, self.phase_update_rate)
            return np.multiply(harmony_memory, np.exp(1j * (self.phase_factor + phase_update) * phase))

        return super().__call__(func)