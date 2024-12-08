import numpy as np

class SimulatedAnnealingPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 10
        self.initial_temperature = 1.0
        self.cooling_rate = 0.95
        self.global_best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        self.global_best_fitness = func(self.global_best_solution)

    def __call__(self, func):
        def acceptance_probability(curr_fitness, new_fitness, temperature):
            if new_fitness < curr_fitness:
                return 1.0
            return np.exp((curr_fitness - new_fitness) / temperature)
        
        current_solution = np.random.uniform(-5.0, 5.0, self.dim)
        current_fitness = func(current_solution)
        best_solution = current_solution
        best_fitness = current_fitness
        temperature = self.initial_temperature
        
        for _ in range(self.budget):
            for _ in range(self.num_particles):
                new_solution = np.clip(current_solution + np.random.uniform(0, 1, self.dim) * (best_solution - current_solution), -5.0, 5.0)
                new_fitness = func(new_solution)
                
                if acceptance_probability(current_fitness, new_fitness, temperature) > np.random.uniform(0, 1):
                    current_solution = new_solution
                    current_fitness = new_fitness
                    
                    if new_fitness < best_fitness:
                        best_solution = new_solution
                        best_fitness = new_fitness
            
            temperature *= self.cooling_rate
        
        return best_solution