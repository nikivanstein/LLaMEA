import numpy as np

class DELF:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.bounds = (-5.0, 5.0)
        self.pop_size, self.F, self.CR = 10, 0.5, 0.9
        self.best_solution = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        self.best_fitness = func(self.best_solution)

    def levy_flight(self, beta=1.5):
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        s = np.random.normal(0, sigma, self.dim)
        u, v = np.random.normal(0, 1, (2, self.dim))
        step = s / np.abs(u) ** (1 / beta)
        levy = 0.01 * step * v
        return levy

    def __call__(self, func):
        for _ in range(self.budget):
            new_population = []
            for _ in range(self.pop_size):
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                mutant = self.best_solution + self.F * (self.best_solution - new_population[a]) + self.levy_flight()
                trial = np.clip(mutant, *self.bounds)
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                new_vector = np.where(crossover_mask, trial, self.best_solution)
                
                new_fitness = func(new_vector)
                if new_fitness < self.best_fitness:
                    self.best_solution, self.best_fitness = new_vector, new_fitness
                    
                new_population.append(new_vector)
            
        return self.best_solution