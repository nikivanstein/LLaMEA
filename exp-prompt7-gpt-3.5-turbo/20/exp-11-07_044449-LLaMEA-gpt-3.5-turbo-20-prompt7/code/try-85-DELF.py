import numpy as np

class DELF:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def levy_flight(self, beta=1.5):
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        s = np.random.normal(0, sigma, self.dim)
        u = np.random.normal(0, 1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = s / np.abs(u) ** (1 / beta)
        levy = 0.01 * step * v
        return levy

    def __call__(self, func):
        pop_size = 10
        F = 0.5
        CR = 0.9
        bounds = (-5.0, 5.0)
        
        best_solution = np.random.uniform(bounds[0], bounds[1], self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            new_population = []
            for _ in range(pop_size):
                a, b, c = np.random.choice(range(pop_size), 3, replace=False)
                mutant = best_solution + F * (best_solution - new_population[a]) + self.levy_flight()
                trial = np.clip(mutant, bounds[0], bounds[1])
                
                crossover_mask = np.random.rand(self.dim) < CR
                new_vector = np.where(crossover_mask, trial, best_solution)
                
                new_fitness = func(new_vector)
                if new_fitness < best_fitness:
                    best_solution = new_vector
                    best_fitness = new_fitness
                    
                new_population.append(new_vector)
            
        return best_solution