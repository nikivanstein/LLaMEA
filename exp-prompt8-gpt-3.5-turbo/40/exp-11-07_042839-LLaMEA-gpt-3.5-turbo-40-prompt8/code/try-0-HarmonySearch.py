import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjustment_rate=0.1, bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjustment_rate = pitch_adjustment_rate
        self.bandwidth = bandwidth

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

        def generate_new_harmony(harmony_memory):
            new_harmony = np.copy(harmony_memory[np.random.randint(0, self.harmony_memory_size)])
            for i in range(self.dim):
                if np.random.rand() < self.pitch_adjustment_rate:
                    new_harmony[i] += np.random.uniform(-self.bandwidth, self.bandwidth)
                    new_harmony[i] = np.clip(new_harmony[i], -5.0, 5.0)
            return new_harmony

        def evaluate_harmony(harmony):
            return func(harmony)

        harmony_memory = initialize_harmony_memory()
        for _ in range(self.budget):
            new_harmony = generate_new_harmony(harmony_memory)
            if evaluate_harmony(new_harmony) < evaluate_harmony(harmony_memory[-1]):
                harmony_memory[-1] = new_harmony
                harmony_memory = harmony_memory[np.argsort([evaluate_harmony(h) for h in harmony_memory])]
        return harmony_memory[0]