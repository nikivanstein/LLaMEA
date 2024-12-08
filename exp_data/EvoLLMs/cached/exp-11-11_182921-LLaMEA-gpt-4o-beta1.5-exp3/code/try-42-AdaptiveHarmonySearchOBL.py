import numpy as np

class AdaptiveHarmonySearchOBL:
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

        # Adaptive parameters
        self.harmony_consideration_rate = 0.9
        self.pitch_adjustment_rate = 0.1
        self.bandwidth = 0.01

    def opposition_based_learning(self, vector):
        return self.lower_bound + self.upper_bound - vector

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            new_harmony = np.copy(self.harmony_memory[np.random.randint(self.harmony_memory_size)])

            # Harmony memory consideration
            if np.random.rand() < self.harmony_consideration_rate:
                for i in range(self.dim):
                    if np.random.rand() < self.pitch_adjustment_rate:
                        new_harmony[i] += np.random.uniform(-1, 1) * self.bandwidth
                        new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)

            # Opposition-based learning
            opposite_harmony = self.opposition_based_learning(new_harmony)

            # Evaluate both new harmony and its opposite
            new_score = func(new_harmony)
            opposite_score = func(opposite_harmony)
            self.func_evaluations += 2

            if new_score < self.best_score or opposite_score < self.best_score:
                if new_score < opposite_score:
                    if new_score < self.best_score:
                        self.best_score = new_score
                        self.best_position = new_harmony
                else:
                    if opposite_score < self.best_score:
                        self.best_score = opposite_score
                        self.best_position = opposite_harmony

            # Insert better harmony into memory
            worst_idx = np.argmax([func(h) for h in self.harmony_memory])
            if new_score < func(self.harmony_memory[worst_idx]):
                self.harmony_memory[worst_idx] = new_harmony

            if opposite_score < func(self.harmony_memory[worst_idx]):
                self.harmony_memory[worst_idx] = opposite_harmony

            # Adaptive adjustment of parameters
            self.harmony_consideration_rate -= 0.0001
            self.pitch_adjustment_rate += 0.0001

        return self.best_position