import numpy as np

class HarmonySearchImproved:
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
            step_size = 0.2  # Dynamic step size adjustment
            for i in range(self.dim):
                perturbed_harmony = np.copy(harmony)
                perturbed_harmony[i] += step_size
                if func(perturbed_harmony) < func(harmony):
                    harmony[i] += step_size
                elif func(perturbed_harmony) > func(harmony):
                    harmony[i] -= step_size
            return harmony

        harmony_memory = initialize_harmony_memory(10)

        for _ in range(self.budget):
            new_harmony = generate_new_harmony(harmony_memory)
            new_harmony = local_search(new_harmony)
            if func(new_harmony) < np.min(func(harmony_memory)):
                idx = np.argmin(func(harmony_memory))
                harmony_memory[idx] = new_harmony

        best_solution = harmony_memory[np.argmin(func(harmony_memory))]
        return best_solution