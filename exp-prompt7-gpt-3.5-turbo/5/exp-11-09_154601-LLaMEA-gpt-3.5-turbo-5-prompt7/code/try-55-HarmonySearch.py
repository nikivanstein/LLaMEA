import numpy as np
from multiprocessing import Pool

class HarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjust_rate=0.1, bandwidth=0.01, num_processes=4):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.bandwidth = bandwidth
        self.num_processes = num_processes

    def evaluate_solution(self, solution, func):
        return func(solution)

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))

        def pitch_adjustment(harmony):
            num_adjust = int(self.pitch_adjust_rate * self.dim)
            indices = np.random.choice(self.dim, num_adjust, replace=False)
            harmony[indices] = harmony[indices] + np.random.uniform(-self.bandwidth, self.bandwidth, size=num_adjust)
            return harmony

        harmony_memory = initialize_harmony_memory()
        while self.budget > 0:
            new_harmony = np.mean(harmony_memory, axis=0)
            new_harmony = pitch_adjustment(new_harmony)

            with Pool(self.num_processes) as p:
                results = p.starmap(self.evaluate_solution, [(candidate, func) for candidate in harmony_memory])

            if func(new_harmony) < min(results):
                replace_idx = np.argmax(results)
                harmony_memory[replace_idx] = new_harmony
                if np.random.rand() < 0.5 and self.harmony_memory_size < 20:
                    self.harmony_memory_size += 1
                    harmony_memory = np.vstack((harmony_memory, np.random.uniform(-5.0, 5.0, size=self.dim)))
                    harmony_memory = np.delete(harmony_memory, replace_idx, axis=0)
            self.budget -= 1

        return harmony_memory.min(axis=0)