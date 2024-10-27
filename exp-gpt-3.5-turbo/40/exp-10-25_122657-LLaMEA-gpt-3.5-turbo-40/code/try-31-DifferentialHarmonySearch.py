import numpy as np

class DifferentialHarmonySearch:
    def __init__(self, budget, dim, memory_consideration=0.9, pitch_adjustment=0.5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.memory_consideration = memory_consideration
        self.pitch_adjustment = pitch_adjustment

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

        def get_fitness(harmony_memory):
            return np.array([func(solution) for solution in harmony_memory])

        def update_harmony_memory(harmony_memory, fitness):
            idx_sorted = np.argsort(fitness)
            best_idx = idx_sorted[0]
            better_idx = idx_sorted[1]
            worst_idx = idx_sorted[-1]
            new_solution = harmony_memory[best_idx] + self.pitch_adjustment * (harmony_memory[better_idx] - harmony_memory[worst_idx])
            harmony_memory[worst_idx] = new_solution
            return harmony_memory

        harmony_memory = initialize_harmony_memory()
        fitness = get_fitness(harmony_memory)

        for _ in range(self.budget - self.budget // 10):
            harmony_memory = update_harmony_memory(harmony_memory, fitness)
            fitness = get_fitness(harmony_memory)

        best_idx = np.argmin(fitness)
        return harmony_memory[best_idx]