import numpy as np

class FireAntOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.step_size = 1.0

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            new_solution = best_solution + np.random.uniform(-self.step_size, self.step_size, self.dim)
            new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
            new_fitness = func(new_solution)
            
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                self.step_size = min(self.step_size * 1.1, 5.0)  # Adjust step size based on fitness improvement
            else:
                self.step_size = max(self.step_size / 1.1, 0.01)  # Reduce step size for exploration
            
        return best_solution