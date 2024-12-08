import numpy as np

class EvolutionaryHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

        def get_fitness(harmony_memory):
            return np.array([func(solution) for solution in harmony_memory])

        def update_harmony_memory(harmony_memory, fitness):
            elite_idx = np.argsort(fitness)[:min(3, len(fitness))]
            for idx in elite_idx:
                new_solution = np.clip(harmony_memory[idx] + np.random.normal(0, 1, self.dim), self.lower_bound, self.upper_bound)
                if func(new_solution) < func(harmony_memory[idx]):
                    harmony_memory[idx] = new_solution
            return harmony_memory

        harmony_memory = initialize_harmony_memory()
        fitness = get_fitness(harmony_memory)

        for _ in range(self.budget - self.budget // 10):
            harmony_memory = update_harmony_memory(harmony_memory, fitness)
            fitness = get_fitness(harmony_memory)

        best_idx = np.argmin(fitness)
        return harmony_memory[best_idx]