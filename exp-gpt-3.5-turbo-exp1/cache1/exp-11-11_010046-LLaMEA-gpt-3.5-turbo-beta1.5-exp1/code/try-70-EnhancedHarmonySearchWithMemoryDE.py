import numpy as np
from scipy.optimize import differential_evolution

class EnhancedHarmonySearchWithMemoryDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = int(0.1 * budget)
        self.bandwidth_min = 0.01
        self.bandwidth_max = 0.1
        self.bandwidth_decay = (self.bandwidth_max - self.bandwidth_min) / (0.5 * budget)  # Dynamic adjustment of bandwidth
        self.local_search_min = 0.05
        self.local_search_max = 0.2
        self.local_search_decay = (self.local_search_max - self.local_search_min) / (0.5 * budget)  # Dynamic adjustment of local search step
        self.harmony_memory = np.random.uniform(-5.0, 5.0, (self.memory_size, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            de_bounds = [(-5.0, 5.0)] * self.dim
            de_solution = differential_evolution(func, de_bounds, strategy='best1bin').x
            self.bandwidth = max(self.bandwidth_min, self.bandwidth_max - self.bandwidth_decay * _)
            new_solution = np.clip(np.random.normal(de_solution, self.bandwidth), -5.0, 5.0)
            if func(new_solution) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_solution
                self.harmony_memory = self.harmony_memory[self.harmony_memory[:, 0].argsort()]
            best_solution = self.harmony_memory[0]
            self.local_search_step = max(self.local_search_min, self.local_search_max - self.local_search_decay * _)
            perturbed_solution = np.clip(best_solution + np.random.uniform(-self.local_search_step, self.local_search_step, self.dim), -5.0, 5.0)
            if func(perturbed_solution) < func(best_solution):
                self.harmony_memory[0] = perturbed_solution
                self.harmony_memory = self.harmony_memory[self.harmony_memory[:, 0].argsort()]
        return self.harmony_memory[0]