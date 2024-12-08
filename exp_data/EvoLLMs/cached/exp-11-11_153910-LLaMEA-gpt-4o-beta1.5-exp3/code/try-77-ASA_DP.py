import numpy as np

class ASA_DP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.temperature = 1.0
        self.cooling_rate = 0.99
        self.best_global_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.best_global_fitness = float('inf')
        self.evaluations = 0

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def perturb(self, position):
        perturbation = np.random.normal(0, 1, self.dim)
        return position + 0.1 * perturbation

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        current_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        current_fitness = self.evaluate(func, current_position)
        
        while self.evaluations < self.budget:
            new_position = self.perturb(current_position)
            new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
            new_fitness = self.evaluate(func, new_position)
            
            if new_fitness < current_fitness or np.random.rand() < np.exp(-(new_fitness - current_fitness) / self.temperature):
                current_position = new_position
                current_fitness = new_fitness
                
                if new_fitness < self.best_global_fitness:
                    self.best_global_fitness = new_fitness
                    self.best_global_position = new_position
            
            self.temperature *= self.cooling_rate
        
        return self.best_global_position