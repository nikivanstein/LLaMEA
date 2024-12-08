import numpy as np

class EvoHarmonySearch:
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
            population_size = len(harmony_memory)
            worst_idx = np.argmax(fitness)
            new_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            harmony_memory[worst_idx] = new_solution
            for idx in range(population_size):
                if np.random.rand() < 0.4:  # Probability to change individual lines
                    harmony_memory[idx] += np.random.normal(0, 0.1, self.dim)  # Introducing mutation
                    harmony_memory[idx] = np.clip(harmony_memory[idx], self.lower_bound, self.upper_bound)
            return harmony_memory

        harmony_memory = initialize_harmony_memory()
        fitness = get_fitness(harmony_memory)

        for _ in range(self.budget - self.budget // 10):
            harmony_memory = update_harmony_memory(harmony_memory, fitness)
            fitness = get_fitness(harmony_memory)

        best_idx = np.argmin(fitness)
        return harmony_memory[best_idx]