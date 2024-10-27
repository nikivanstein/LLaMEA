import numpy as np

class EvolutionaryCrossoverOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def create_solution():
            return np.random.uniform(-5.0, 5.0, self.dim)

        def novel_crossover(solution, best_solution):
            return solution + np.random.uniform(-1, 1, self.dim) * (best_solution - solution)

        solutions = [create_solution() for _ in range(self.budget)]
        best_solution = solutions[0]  # Initialize with the first solution
        for _ in range(self.budget):
            for i in range(len(solutions)):
                new_solution = novel_crossover(solutions[i], best_solution)
                if func(new_solution) < func(solutions[i]):
                    solutions[i] = new_solution
                if func(solutions[i]) < func(best_solution):
                    best_solution = solutions[i]

        return best_solution