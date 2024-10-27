import numpy as np

class ImprovedHarmonySearch:
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

        def differential_evolution(harmony_memory, fitness, f=0.5, cr=0.9):
            best_idx = np.argmin(fitness)
            best_solution = harmony_memory[best_idx]
            for i in range(self.budget // 10):
                idxs = np.random.choice(len(harmony_memory), 3, replace=False)
                a, b, c = harmony_memory[idxs]
                mutant = np.clip(a + f * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, harmony_memory[i])
                if func(trial) < func(harmony_memory[i]):
                    harmony_memory[i] = trial
            return harmony_memory

        harmony_memory = initialize_harmony_memory()
        fitness = get_fitness(harmony_memory)

        for _ in range(self.budget - self.budget // 10):
            harmony_memory = update_harmony_memory(harmony_memory, fitness)
            fitness = get_fitness(harmony_memory)
            harmony_memory = differential_evolution(harmony_memory, fitness)

        best_idx = np.argmin(fitness)
        return harmony_memory[best_idx]