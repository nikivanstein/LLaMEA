import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ParallelHarmonySearchImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def initialize_harmony_memory(size):
            return np.random.uniform(-5.0, 5.0, (size, self.dim))

        def generate_new_harmony(harmony_memory):
            new_harmony = np.random.uniform(-5.0, 5.0, self.dim)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    new_harmony[i] = np.random.choice(harmony_memory[:, i])
            return new_harmony

        def local_search(harmony):
            step_size = 0.1
            for i in range(self.dim):
                perturbed_harmony = np.copy(harmony)
                perturbed_harmony[i] += step_size
                if func(perturbed_harmony) < func(harmony):
                    harmony[i] += step_size
                elif func(perturbed_harmony) > func(harmony):
                    harmony[i] -= step_size
            return harmony

        harmony_memory = initialize_harmony_memory(10)

        with ThreadPoolExecutor() as executor:
            for _ in range(self.budget):
                new_harmonies = list(executor.map(generate_new_harmony, [harmony_memory]*10))
                new_harmonies = list(executor.map(local_search, new_harmonies))
                new_values = list(executor.map(func, new_harmonies))
                min_idx = np.argmin(new_values)
                if new_values[min_idx] < np.min(func(harmony_memory)):
                    harmony_memory[np.argmin(func(harmony_memory))] = new_harmonies[min_idx]

        best_solution = harmony_memory[np.argmin(func(harmony_memory))]
        return best_solution