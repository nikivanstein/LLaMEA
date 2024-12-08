import numpy as np

class AdaptiveHarmonySearchQuantumPitch:
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
        self.harmony_memory_consideration_rate = 0.95
        self.pitch_adjustment_rate = 0.7
        self.bw = 0.01  # Bandwidth for pitch adjustment

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            new_harmonies = np.copy(self.harmony_memory)
            for i in range(self.harmony_memory_size):
                new_harmony = np.copy(new_harmonies[i])
                for j in range(self.dim):
                    if np.random.rand() < self.harmony_memory_consideration_rate:
                        new_harmony[j] = self.harmony_memory[np.random.choice(self.harmony_memory_size), j]
                        if np.random.rand() < self.pitch_adjustment_rate:
                            new_harmony[j] += self.bw * np.random.normal(0, 1)
                    else:
                        new_harmony[j] = np.random.uniform(self.lower_bound, self.upper_bound)

                new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
                new_score = func(new_harmony)
                self.func_evaluations += 1

                if new_score < func(new_harmonies[i]):
                    new_harmonies[i] = new_harmony
                    if new_score < self.best_score:
                        self.best_score = new_score
                        self.best_position = new_harmony

            self.harmony_memory = new_harmonies

            # Adaptive adjustment of harmony memory consideration rate and bandwidth
            self.harmony_memory_consideration_rate = 0.95 - 0.45 * (self.func_evaluations / self.budget)
            self.pitch_adjustment_rate = 0.7 * (1 + 0.3 * np.cos(2 * np.pi * self.func_evaluations / self.budget))
            self.bw = 0.01 + 0.04 * np.sin(2 * np.pi * self.func_evaluations / self.budget)

        return self.best_position