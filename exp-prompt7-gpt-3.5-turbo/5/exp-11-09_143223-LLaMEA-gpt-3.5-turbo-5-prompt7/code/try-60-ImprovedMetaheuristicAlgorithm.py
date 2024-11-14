import numpy as np

class ImprovedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_step = 0.5  # Initial mutation step size
        self.learning_rate = 0.1

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            candidate_solution = best_solution + self.mutation_step * np.random.uniform(-1, 1, self.dim)
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
            
            # Dynamic learning rate adjustment
            if np.random.rand() < 0.1:  # Probability of step size adjustment
                if candidate_fitness < best_fitness:
                    self.mutation_step *= np.exp(-self.learning_rate)
                else:
                    self.mutation_step *= np.exp(self.learning_rate)
                self.mutation_step = max(0.1, min(self.mutation_step, 2.0))

        return best_solution