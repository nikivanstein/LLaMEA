class FireAntOptimizationImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        step_size = 1.0
        
        for _ in range(self.budget):
            new_solution = best_solution + np.random.uniform(-step_size, step_size, self.dim)
            new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
            new_fitness = func(new_solution)
            
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                step_size *= 0.95  # Decrease step size for exploitation
            else:
                step_size *= 1.02  # Increase step size for exploration
        
        return best_solution