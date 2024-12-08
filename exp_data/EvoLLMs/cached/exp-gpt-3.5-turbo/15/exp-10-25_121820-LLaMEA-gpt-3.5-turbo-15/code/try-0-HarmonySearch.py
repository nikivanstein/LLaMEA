import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=20, pitch_adjust_rate=0.1, bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.bandwidth = bandwidth

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

        def pitch_adjustment(harmony):
            selected_idx = np.random.randint(self.dim)
            if np.random.rand() < self.pitch_adjust_rate:
                harmony[selected_idx] = np.clip(harmony[selected_idx] + np.random.uniform(-self.bandwidth, self.bandwidth), -5.0, 5.0)
            return harmony

        harmony_memory = initialize_harmony_memory()
        harmony_costs = np.array([func(harmony) for harmony in harmony_memory])
        for _ in range(self.budget - self.harmony_memory_size):
            new_harmony = pitch_adjustment(np.copy(harmony_memory[np.argmin(harmony_costs)]))
            new_cost = func(new_harmony)
            if new_cost < np.max(harmony_costs):
                replace_idx = np.argmax(harmony_costs)
                harmony_memory[replace_idx] = new_harmony
                harmony_costs[replace_idx] = new_cost

        best_idx = np.argmin(harmony_costs)
        return harmony_memory[best_idx]