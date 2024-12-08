import numpy as np

class ImprovedAdaptiveFireflyAlgorithmOptimized:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def attractiveness(distance):
            return np.exp(-distance)

        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            current_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            current_fitness = func(current_solution)

            if current_fitness < best_fitness:
                best_solution = current_solution
                best_fitness = current_fitness

            for _ in range(self.budget):
                if func(current_solution) < func(best_solution):
                    best_solution = current_solution

                step_size = np.random.uniform(0, 0.1, self.dim)  # Exploration
                current_solution = current_solution * 0.9 + best_solution * 0.1 + step_size
                
                for i in range(self.dim):  # Exploitation
                    current_solution[i] = np.clip(current_solution[i], self.lower_bound, self.upper_bound)

                new_solution = current_solution * (1 - attractiveness(np.linalg.norm(current_solution - best_solution))) + best_solution * attractiveness(np.linalg.norm(current_solution - best_solution))
                if func(new_solution) < func(current_solution):
                    current_solution = new_solution
        return best_solution