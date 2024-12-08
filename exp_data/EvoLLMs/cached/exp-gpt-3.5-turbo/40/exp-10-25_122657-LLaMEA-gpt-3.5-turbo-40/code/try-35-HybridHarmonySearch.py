import numpy as np

class HybridHarmonySearch:
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
            worst_idx = np.argmax(fitness)
            new_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            harmony_memory[worst_idx] = new_solution
            return harmony_memory

        def apply_differential_evolution(harmony_memory, fitness):
            crossover_rate = 0.9
            scale_factor = 0.5
            for i in range(self.budget // 10):
                idxs = np.random.choice(self.budget, 3, replace=False)
                mutant = harmony_memory[idxs[0]] + scale_factor * (harmony_memory[idxs[1]] - harmony_memory[idxs[2]])
                trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, harmony_memory[i])
                if func(trial) < fitness[i]:
                    harmony_memory[i] = trial
                    fitness[i] = func(trial)
            return harmony_memory

        harmony_memory = initialize_harmony_memory()
        fitness = get_fitness(harmony_memory)

        for _ in range(self.budget - self.budget // 10):
            harmony_memory = update_harmony_memory(harmony_memory, fitness)
            fitness = get_fitness(harmony_memory)

        harmony_memory = apply_differential_evolution(harmony_memory, fitness)

        best_idx = np.argmin(fitness)
        return harmony_memory[best_idx]