import numpy as np
from multiprocessing import Pool

class ParallelHarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjust_rate=0.1, bandwidth=0.01, num_processes=4):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.bandwidth = bandwidth
        self.num_processes = num_processes

    def evaluate_func(self, harmony, func):
        return func(harmony)

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))

        def pitch_adjustment(harmony):
            num_adjust = int(self.pitch_adjust_rate * self.dim)
            indices = np.random.choice(self.dim, num_adjust, replace=False)
            harmony[indices] = harmony[indices] + np.random.uniform(-self.bandwidth, self.bandwidth, size=num_adjust)
            return harmony

        harmony_memory = initialize_harmony_memory()
        with Pool(self.num_processes) as pool:
            while self.budget > 0:
                harmonies = [np.mean(harmony_memory, axis=0) for _ in range(self.num_processes)]
                new_harmonies = pool.starmap(pitch_adjustment, [(h,) for h in harmonies])
                evaluations = pool.starmap(self.evaluate_func, [(h, func) for h in new_harmonies])
                min_idx = np.argmin(evaluations)
                if evaluations[min_idx] < func(harmony_memory.min(axis=0)):
                    harmony_memory[np.argmax(func(harmony_memory))] = new_harmonies[min_idx]
                    if np.random.rand() < 0.5 and self.harmony_memory_size < 20:
                        self.harmony_memory_size += 1
                        harmony_memory = np.vstack((harmony_memory, np.random.uniform(-5.0, 5.0, size=self.dim)))
                        harmony_memory = np.delete(harmony_memory, np.argmax(func(harmony_memory)), axis=0)
                self.budget -= 1

        return harmony_memory.min(axis=0)