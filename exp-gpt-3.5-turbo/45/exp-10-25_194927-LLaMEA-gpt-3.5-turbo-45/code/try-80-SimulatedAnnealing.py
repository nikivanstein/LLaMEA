import numpy as np

class SimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_temperature = 10.0
        self.final_temperature = 0.1
        self.adaptive_cooling_rate = 1.0 - (self.final_temperature / self.initial_temperature)

    def __call__(self, func):
        current_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_solution = current_solution
        temperature = self.initial_temperature

        for _ in range(self.budget):
            new_solution = current_solution + np.random.uniform(-1, 1, self.dim) * temperature
            new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
            
            current_fitness = func(current_solution)
            new_fitness = func(new_solution)

            if new_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - new_fitness) / temperature):
                current_solution = new_solution
                if new_fitness < func(best_solution):
                    best_solution = new_solution
            
            temperature *= self.adaptive_cooling_rate
        
        return best_solution