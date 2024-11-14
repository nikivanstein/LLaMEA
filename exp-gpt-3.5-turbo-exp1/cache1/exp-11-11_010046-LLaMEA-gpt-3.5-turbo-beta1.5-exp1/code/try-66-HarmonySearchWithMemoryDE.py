import numpy as np
from scipy.optimize import differential_evolution

class HarmonySearchWithMemoryDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = int(0.1 * budget)
        self.bandwidth = 0.02  # Increased bandwidth for broader exploration
        self.local_search_step = 0.1  # Define step size for local search
        self.harmony_memory = np.random.uniform(-5.0, 5.0, (self.memory_size, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            de_bounds = [(-5.0, 5.0)] * self.dim
            de_solution = differential_evolution(func, de_bounds, strategy='best1bin').x
            new_solution = np.clip(np.random.normal(de_solution, self.bandwidth), -5.0, 5.0)
            if func(new_solution) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_solution
                self.harmony_memory = self.harmony_memory[self.harmony_memory[:,0].argsort()]
            best_solution = self.harmony_memory[0]
            improvement_ratio = 1.0 - func(best_solution) / func(self.harmony_memory[-1])
            adaptive_step_size = self.local_search_step * (1 + improvement_ratio)
            perturbed_solution = np.clip(best_solution + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), -5.0, 5.0)
            if func(perturbed_solution) < func(best_solution):
                self.harmony_memory[0] = perturbed_solution
                self.harmony_memory = self.harmony_memory[self.harmony_memory[:,0].argsort()]
        return self.harmony_memory[0]