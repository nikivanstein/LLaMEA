import numpy as np

class EnhancedAdaptiveFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.attractiveness = lambda distance: np.exp(-distance)

    def __call__(self, func):
        best_solution = np.random.uniform(*self.bounds, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            current_solution = np.random.uniform(*self.bounds, self.dim)
            current_fitness = func(current_solution)

            if current_fitness < best_fitness:
                best_solution, best_fitness = current_solution, current_fitness

            for _ in range(self.budget):
                if func(current_solution) < func(best_solution):
                    best_solution = current_solution

                step_size = np.random.uniform(0, 0.1, self.dim)
                current_solution = 0.9 * current_solution + 0.1 * best_solution + step_size

                current_solution = np.clip(current_solution, *self.bounds)

                new_solution = (1 - self.attractiveness(np.linalg.norm(current_solution - best_solution))) * current_solution + self.attractiveness(np.linalg.norm(current_solution - best_solution)) * best_solution
                if func(new_solution) < func(current_solution):
                    current_solution = new_solution
        return best_solution