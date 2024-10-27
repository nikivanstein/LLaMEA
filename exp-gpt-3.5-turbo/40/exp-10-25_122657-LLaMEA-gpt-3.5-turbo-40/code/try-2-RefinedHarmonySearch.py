import numpy as np

class RefinedHarmonySearch:
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

        def refine_solution(solution, best_solution, f=0.5, cr=0.9):
            mutated_solution = solution + f * (best_solution - solution)
            crossover = np.random.rand(self.dim) < cr
            trial_solution = np.where(crossover, mutated_solution, solution)
            return trial_solution

        harmony_memory = initialize_harmony_memory()
        fitness = get_fitness(harmony_memory)

        for _ in range(self.budget - self.budget // 10):
            best_idx = np.argmin(fitness)
            best_solution = harmony_memory[best_idx]

            for i in range(len(harmony_memory)):
                harmony_memory[i] = refine_solution(harmony_memory[i], best_solution)

            fitness = get_fitness(harmony_memory)

        best_idx = np.argmin(fitness)
        return harmony_memory[best_idx]