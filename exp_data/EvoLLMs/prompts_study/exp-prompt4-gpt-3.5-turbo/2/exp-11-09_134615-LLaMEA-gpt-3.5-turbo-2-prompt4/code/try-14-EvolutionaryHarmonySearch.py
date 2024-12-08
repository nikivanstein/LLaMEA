import numpy as np

class EvolutionaryHarmonySearch:
    def __init__(self, budget, dim, hm_size=20, hm_rate=0.7, hm_accept=0.1, pa_rate=0.45, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hm_size = hm_size
        self.hm_rate = hm_rate
        self.hm_accept = hm_accept
        self.pa_rate = pa_rate
        self.bw = bw

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.hm_size, self.dim))

        def explore_new_solution(harmony_memory):
            new_solution = np.clip(harmony_memory[np.random.randint(self.hm_size)] + np.random.uniform(-self.bw, self.bw, self.dim),
                                    -5.0, 5.0)
            return new_solution

        def update_harmony_memory(harmony_memory, new_solution, fitness_values):
            worst_index = np.argmax(fitness_values)
            if fitness_values[worst_index] > func(harmony_memory[worst_index]):
                harmony_memory[worst_index] = new_solution
            return harmony_memory

        harmony_memory = initialize_harmony_memory()
        fitness_values = np.array([func(hm) for hm in harmony_memory])

        for _ in range(self.budget - self.hm_size):
            new_solution = explore_new_solution(harmony_memory)
            new_fitness = func(new_solution)
            if new_fitness < np.max(fitness_values):
                harmony_memory = update_harmony_memory(harmony_memory, new_solution, fitness_values)
                fitness_values = np.array([func(hm) for hm in harmony_memory])

        return harmony_memory[np.argmin(fitness_values)]