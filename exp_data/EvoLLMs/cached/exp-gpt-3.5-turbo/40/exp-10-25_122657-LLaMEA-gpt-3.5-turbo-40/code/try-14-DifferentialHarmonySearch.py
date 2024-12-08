import numpy as np

class DifferentialHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.p_crossover = 0.5
        self.p_mutation = 0.1
        self.f_scale = 0.5

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

        def get_fitness(harmony_memory):
            return np.array([func(solution) for solution in harmony_memory])

        def update_harmony_memory(harmony_memory, fitness):
            new_harmony_memory = np.copy(harmony_memory)
            for i in range(self.budget):
                r = np.random.rand()
                if r < self.p_crossover:
                    idxs = np.random.choice(self.budget, 3, replace=False)
                    differential_vector = new_harmony_memory[idxs[0]] + self.f_scale * (new_harmony_memory[idxs[1]] - new_harmony_memory[idxs[2]])
                    crossover_mask = np.random.rand(self.dim) < 0.5
                    new_harmony_memory[i] = np.where(crossover_mask, differential_vector, new_harmony_memory[i])
                r = np.random.rand()
                if r < self.p_mutation:
                    new_harmony_memory[i] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            return new_harmony_memory

        harmony_memory = initialize_harmony_memory()
        fitness = get_fitness(harmony_memory)

        for _ in range(self.budget - self.budget // 10):
            harmony_memory = update_harmony_memory(harmony_memory, fitness)
            fitness = get_fitness(harmony_memory)

        best_idx = np.argmin(fitness)
        return harmony_memory[best_idx]