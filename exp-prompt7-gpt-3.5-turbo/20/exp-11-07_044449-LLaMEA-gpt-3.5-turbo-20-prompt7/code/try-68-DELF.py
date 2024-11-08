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
            new_population = [best_solution + F * (best_solution - new_population[np.random.choice(range(pop_size), 1)[0]]) + self.levy_flight() for _ in range(pop_size)]
            new_population = np.clip(new_population, bounds[0], bounds[1])
            
            fitness_values = np.array([func(ind) for ind in new_population])
            min_idx = np.argmin(fitness_values)
            
            if fitness_values[min_idx] < best_fitness:
                best_solution = new_population[min_idx]
                best_fitness = fitness_values[min_idx]
                    
        return best_solution