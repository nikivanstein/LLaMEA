import numpy as np

class EnhancedOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def init_candidate():
            return np.random.uniform(-5.0, 5.0, size=(self.dim,))
        
        def levy_flight(step_size):
            beta = 1.5
            sigma = (np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.normal(0, sigma, size=(self.dim,))
            v = np.random.normal(0, 1, size=(self.dim,))
            step = u / (np.abs(v) ** (1 / beta))
            return step_size * step
        
        best_solution = init_candidate()
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            for _ in range(5):
                candidates = [init_candidate() for _ in range(10)]
                for candidate in candidates:
                    step_size = np.random.uniform(0.1, 1.0)
                    new_candidate = candidate + levy_flight(step_size)
                    new_fitness = func(new_candidate)
                    if new_fitness < best_fitness:
                        best_solution = new_candidate
                        best_fitness = new_fitness

                # Differential Evolution
                mutant_candidate = best_solution + 0.5 * (candidates[0] - candidates[1]) + 0.5 * (candidates[2] - candidates[3])
                mutant_fitness = func(mutant_candidate)
                if mutant_fitness < best_fitness:
                    best_solution = mutant_candidate
                    best_fitness = mutant_fitness
                    candidates[4] = mutant_candidate

                # Adaptive Mutation
                step_size = np.random.uniform(0.1, 1.0)
                new_candidate = best_solution + levy_flight(step_size)
                new_fitness = func(new_candidate)
                if new_fitness < best_fitness:
                    best_solution = new_candidate
                    best_fitness = new_fitness

                # Line Refinement with probability 0.35
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