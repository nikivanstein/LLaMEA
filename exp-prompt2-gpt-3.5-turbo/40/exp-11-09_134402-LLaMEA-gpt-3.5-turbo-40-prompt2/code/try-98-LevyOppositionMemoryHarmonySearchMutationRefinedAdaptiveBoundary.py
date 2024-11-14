import numpy as np

class LevyOppositionMemoryHarmonySearchMutationRefinedAdaptiveBoundary:
    def __init__(self, budget, dim, harmony_memory_size=10, bandwidth=0.01, bandwidth_range=(0.01, 0.1), pitch_adjustment_rate=0.2, pitch_adjustment_range=(0.1, 0.5), memory_consideration_prob=0.5, dynamic_memory_prob_range=(0.4, 0.8), mutation_rate=0.1, opposition_rate=0.5, adaptive_boundary_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.bandwidth_range = bandwidth_range
        self.pitch_adjustment_rate = pitch_adjustment_rate
        self.pitch_adjustment_range = pitch_adjustment_range
        self.memory_consideration_prob = memory_consideration_prob
        self.dynamic_memory_prob_range = dynamic_memory_prob_range
        self.mutation_rate = mutation_rate
        self.opposition_rate = opposition_rate
        self.adaptive_boundary_rate = adaptive_boundary_rate

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

        def levy_flight():
            beta = 1.5
            alpha = 0.01 * np.power(1 / beta, (1 / beta))
            u = np.random.normal(0, alpha, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.power(np.abs(v), 1 / beta)
            return step

        def update_harmony_memory(harmony_memory, new_solution):
            harmony_memory = np.vstack((harmony_memory, new_solution))
            harmony_memory = harmony_memory[np.argsort(func(harmony_memory))]
            return harmony_memory[:self.harmony_memory_size]

        def improvise(harmony_memory):
            new_solution = np.copy(harmony_memory[np.random.randint(self.harmony_memory_size)])
            for i in range(self.dim):
                if np.random.rand() < self.bandwidth:
                    new_solution[i] = np.random.uniform(-5.0, 5.0)
                if np.random.rand() < self.pitch_adjustment_rate:
                    pitch_range = np.random.uniform(*self.pitch_adjustment_range)
                    new_solution[i] += np.random.uniform(-pitch_range, pitch_range)
                    new_solution[i] = np.clip(new_solution[i], -5.0, 5.0)
                if np.random.rand() < np.random.uniform(*self.dynamic_memory_prob_range):
                    new_solution[i] = harmony_memory[np.random.randint(self.harmony_memory_size), i]
                if np.random.rand() < self.mutation_rate:
                    new_solution[i] = np.random.uniform(-5.0, 5.0)
                if np.random.rand() < np.random.uniform(0.3, 0.7):
                    new_solution[i] = 2 * np.mean(harmony_memory[:, i]) - new_solution[i]
                if np.random.rand() < self.adaptive_boundary_rate:
                    boundary_shift = np.random.uniform(-0.1, 0.1)
                    new_solution[i] = np.clip(new_solution[i] + boundary_shift, -5.0, 5.0)
                new_solution += levy_flight()
            return new_solution

        harmony_memory = initialize_harmony_memory()
        for _ in range(self.budget):
            self.bandwidth = np.clip(self.bandwidth + np.random.uniform(-0.01, 0.01), *self.bandwidth_range)
            self.pitch_adjustment_rate = np.clip(self.pitch_adjustment_rate + np.random.uniform(-0.05, 0.05),
                                                  *self.pitch_adjustment_range)
            new_solution = improvise(harmony_memory)
            if func(new_solution) < func(harmony_memory[-1]):
                harmony_memory = update_harmony_memory(harmony_memory, new_solution)

        return harmony_memory[0]
