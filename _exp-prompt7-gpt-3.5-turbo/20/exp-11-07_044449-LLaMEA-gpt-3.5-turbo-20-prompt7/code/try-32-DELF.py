import numpy as np

class DELF:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.pop_size = 10
        self.F = 0.5
        self.CR = 0.9

    def levy_flight(self, beta=1.5):
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        s = np.random.normal(0, sigma, self.dim)
        u = np.random.normal(0, 1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = s / np.abs(u) ** (1 / beta)
        levy = 0.01 * step * v
        return levy

    def __call__(self, func):
        best_solution = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            new_population = [best_solution]
            for _ in range(1, self.pop_size):
                a, b, c = np.random.choice(range(len(new_population)), 3, replace=False)
                mutant = best_solution + self.F * (best_solution - new_population[a]) + self.levy_flight()
                trial = np.clip(mutant, self.bounds[0], self.bounds[1])
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                new_vector = np.where(crossover_mask, trial, best_solution)
                
                new_fitness = func(new_vector)
                if new_fitness < best_fitness:
                    best_solution = new_vector
                    best_fitness = new_fitness
                    
                new_population.append(new_vector)
            
        return best_solution
