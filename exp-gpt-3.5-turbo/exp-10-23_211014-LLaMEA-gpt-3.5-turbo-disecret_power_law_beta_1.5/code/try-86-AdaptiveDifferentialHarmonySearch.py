import numpy as np

class AdaptiveDifferentialHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.harmony_memory = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.harmony_memory_size = 100
        self.bandwidth = 0.02
        self.pitch_adjustment_rate = 0.5
        self.diversity_rate = 0.7
        self.convergence_threshold = 0.1
        self.convergence_counter = 0
        self.best_solution = None

    def __call__(self, func):
        for _ in range(self.budget):
            harmony = np.random.uniform(-5.0, 5.0, self.dim)
            fitness = func(harmony)
            if self.best_solution is None or fitness < func(self.best_solution):
                self.best_solution = harmony
            if self.convergence_counter > self.convergence_threshold * self.budget:
                self.adapt_strategy()
            self.update_harmony_memory(harmony, fitness)
        return self.best_solution

    def adapt_strategy(self):
        self.bandwidth *= 0.9
        self.pitch_adjustment_rate *= 0.9
        self.diversity_rate *= 1.1
        self.convergence_counter = 0

    def update_harmony_memory(self, harmony, fitness):
        worst_idx = np.argmax([func(h) for h in self.harmony_memory])
        if fitness < func(self.harmony_memory[worst_idx]):
            self.harmony_memory[worst_idx] = harmony

        self.convergence_counter += 1