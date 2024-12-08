import numpy as np
from concurrent.futures import ThreadPoolExecutor

class HarmonySearch:
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

        harmony_memory = initialize_harmony_memory(10)

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(func, candidate): candidate for candidate in harmony_memory}
            for future in futures:
                result = future.result()
                if result < np.min([func(h) for h in harmony_memory]):
                    idx = np.argmin([func(h) for h in harmony_memory])
                    harmony_memory[idx] = futures[future]

        best_solution = harmony_memory[np.argmin([func(h) for h in harmony_memory])]
        return best_solution