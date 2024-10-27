import numpy as np

class HarmonySearchSolver:
    def __init__(self, budget, dim, harmony_memory_size=20, band_width=0.01, pitch_adjust_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.band_width = band_width
        self.pitch_adjust_rate = pitch_adjust_rate

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def explore_new_solution(harmony_memory):
            new_solution = np.clip(harmony_memory[np.random.randint(0, len(harmony_memory))] +
                                    np.random.uniform(-self.band_width, self.band_width, size=self.dim), -5.0, 5.0)
            return new_solution

        harmony_memory = np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))
        harmony_costs = np.array([objective_function(x) for x in harmony_memory])
        best_idx = np.argmin(harmony_costs)
        best_harmony = harmony_memory[best_idx]

        for _ in range(self.budget - self.harmony_memory_size):
            new_solution = explore_new_solution(harmony_memory)
            new_solution_cost = objective_function(new_solution)
            if new_solution_cost < harmony_costs[best_idx]:
                harmony_memory[best_idx] = new_solution
                harmony_costs[best_idx] = new_solution_cost
                best_idx = np.argmax(harmony_costs)
                best_harmony = harmony_memory[best_idx]

            for i in range(self.dim):
                if np.random.rand() < self.pitch_adjust_rate:
                    harmony_memory[i, :] = np.clip(harmony_memory[i, :] + np.random.uniform(-self.band_width, self.band_width), -5.0, 5.0)

        return best_harmony