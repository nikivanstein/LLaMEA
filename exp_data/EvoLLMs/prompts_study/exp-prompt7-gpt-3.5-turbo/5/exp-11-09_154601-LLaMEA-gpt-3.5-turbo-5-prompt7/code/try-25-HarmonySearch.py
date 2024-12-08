import numpy as np
from concurrent.futures import ThreadPoolExecutor

class HarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjust_rate=0.1, bandwidth=0.01, num_threads=4):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.bandwidth = bandwidth
        self.num_threads = num_threads

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))

        def pitch_adjustment(harmony):
            num_adjust = int(self.pitch_adjust_rate * self.dim)
            indices = np.random.choice(self.dim, num_adjust, replace=False)
            harmony[indices] = harmony[indices] + np.random.uniform(-self.bandwidth, self.bandwidth, size=num_adjust)
            return harmony

        harmony_memory = initialize_harmony_memory()
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            while self.budget > 0:
                harmonies = [np.mean(harmony_memory, axis=0) for _ in range(self.num_threads)]
                harmonies = list(executor.map(pitch_adjustment, harmonies))
                results = list(executor.map(func, harmonies))
                for idx, harmony in enumerate(harmonies):
                    if results[idx] < func(harmony_memory.min(axis=0)):
                        replace_idx = np.argmax(func(harmony_memory))
                        harmony_memory[replace_idx] = harmony
                        if np.random.rand() < 0.5 and self.harmony_memory_size < 20:
                            self.harmony_memory_size += 1
                            harmony_memory = np.vstack((harmony_memory, np.random.uniform(-5.0, 5.0, size=self.dim)))
                            harmony_memory = np.delete(harmony_memory, replace_idx, axis=0)
                        self.budget -= 1

        return harmony_memory.min(axis=0)