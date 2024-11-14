import numpy as np

class HarmonyDifferentialSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hmcr = 0.9  # Harmony memory consideration rate
        self.par = 0.3  # Pitch adjustment rate
        self.bandwidth = 0.05
        self.evaluations = 0

    def __call__(self, func):
        # Initialize harmony memory with random solutions
        harmony_memory = self.lower_bound + np.random.rand(self.harmony_memory_size, self.dim) * (self.upper_bound - self.lower_bound)
        harmony_values = np.apply_along_axis(func, 1, harmony_memory)
        self.evaluations = self.harmony_memory_size

        best_index = np.argmin(harmony_values)
        best_harmony = harmony_memory[best_index]
        best_value = harmony_values[best_index]

        while self.evaluations < self.budget:
            new_harmony = np.empty(self.dim)

            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    # Memory consideration
                    new_harmony[i] = harmony_memory[np.random.randint(self.harmony_memory_size), i]

                    # Pitch adjustment
                    if np.random.rand() < self.par:
                        new_harmony[i] += self.bandwidth * (np.random.rand() * 2 - 1)  # Adjust within bandwidth
                else:
                    # Random selection
                    new_harmony[i] = self.lower_bound + np.random.rand() * (self.upper_bound - self.lower_bound)

            # Ensure within bounds
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            fitness = func(new_harmony)
            self.evaluations += 1

            # Replace worst harmony if new harmony is better
            worst_index = np.argmax(harmony_values)
            if fitness < harmony_values[worst_index]:
                harmony_memory[worst_index] = new_harmony
                harmony_values[worst_index] = fitness

                # Update best harmony
                if fitness < best_value:
                    best_harmony = new_harmony
                    best_value = fitness

            # Early stopping if budget is exhausted
            if self.evaluations >= self.budget:
                break

        return best_harmony, best_value