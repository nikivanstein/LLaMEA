import numpy as np

class QuantumHarmonySearch:
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
            new_harmony_memory = np.zeros_like(harmony_memory)
            for i in range(self.budget):
                selected_indices = np.random.choice(self.budget, 2, replace=False)
                selected_solutions = harmony_memory[selected_indices]
                updated_solution = np.mean(selected_solutions, axis=0)
                mutation_factor = np.random.uniform(0.01, 0.1)
                mutation_mask = np.random.choice([0, 1], size=self.dim, p=[0.6, 0.4])
                mutation = np.random.uniform(-1, 1, self.dim) * mutation_factor * mutation_mask
                new_harmony = np.clip(updated_solution + mutation, self.lower_bound, self.upper_bound)
                new_harmony_memory[i] = new_harmony
            return new_harmony_memory

        harmony_memory = initialize_harmony_memory()
        fitness = get_fitness(harmony_memory)

        for _ in range(self.budget - self.budget // 10):
            harmony_memory = update_harmony_memory(harmony_memory, fitness)
            fitness = get_fitness(harmony_memory)

        best_idx = np.argmin(fitness)
        return harmony_memory[best_idx]