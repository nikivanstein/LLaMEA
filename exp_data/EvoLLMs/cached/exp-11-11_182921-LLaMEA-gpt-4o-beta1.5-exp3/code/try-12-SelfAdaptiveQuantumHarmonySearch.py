import numpy as np

class SelfAdaptiveQuantumHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = int(10 + 2 * np.sqrt(dim))
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, dim))
        self.func_evaluations = 0
        self.best_score = float('inf')
        self.best_position = None
        self.hmcr = 0.9  # Harmony memory consideration rate
        self.par = 0.3   # Pitch adjustment rate

    def __call__(self, func):
        # Initialize harmony memory using quantum wave function
        self.best_position = self.quantum_wave_initialization(func)

        while self.func_evaluations < self.budget:
            new_harmony = np.copy(self.best_position)

            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[i] = np.random.choice(self.harmony_memory[:, i])
                if np.random.rand() < self.par:
                    new_harmony[i] += np.random.uniform(-1, 1) * (self.upper_bound - self.lower_bound) * 0.1

                new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)

            new_score = func(new_harmony)
            self.func_evaluations += 1

            if new_score < self.best_score:
                self.best_score = new_score
                self.best_position = new_harmony
                worst_index = np.argmax([func(h) for h in self.harmony_memory])
                self.harmony_memory[worst_index] = new_harmony

            # Self-adaptive adjustment of HMCR and PAR
            self.hmcr = 0.9 - 0.8 * (self.func_evaluations / self.budget)
            self.par = 0.1 + 0.2 * (self.func_evaluations / self.budget)

        return self.best_position

    def quantum_wave_initialization(self, func):
        wave_position = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        wave_amplitude = (self.upper_bound - self.lower_bound) / 2.0
        wave_phase = np.random.uniform(0, 2 * np.pi, (self.harmony_memory_size, self.dim))
        wave_function = wave_position + wave_amplitude * np.sin(wave_phase)

        best_wave_score = float('inf')
        best_wave_position = None

        for pos in wave_function:
            score = func(pos)
            if score < best_wave_score:
                best_wave_score = score
                best_wave_position = pos

        self.func_evaluations += self.harmony_memory_size
        self.harmony_memory = wave_function
        return best_wave_position