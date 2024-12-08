import numpy as np

class HarmonySearchDynamicLineRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def initialize_harmonies():
            return np.random.uniform(-5.0, 5.0, size=(self.dim,))

        best_solution = initialize_harmonies()
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            for _ in range(5):
                harmonies = [initialize_harmonies() for _ in range(10)]
                for harmony in harmonies:
                    new_harmony = harmony + np.random.uniform(-1, 1, size=(self.dim,))
                    new_fitness = func(new_harmony)
                    if new_fitness < best_fitness:
                        best_solution = new_harmony
                        best_fitness = new_fitness
                
                # Dynamic Line Refinement with probability 0.35
                if np.random.uniform() < 0.35:
                    line_direction = np.random.uniform(-1, 1, size=(self.dim,))
                    line_direction /= np.linalg.norm(line_direction)
                    line_length = np.random.uniform(0.1, 1.0)
                    line_point = best_solution + line_length * line_direction
                    line_fitness = func(line_point)
                    if line_fitness < best_fitness:
                        best_solution = line_point
                        best_fitness = line_fitness

        return best_solution