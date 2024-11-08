import numpy as np

class DELF:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def levy_flight(self, beta=1.5):
        sigma = (np.exp(np.lgamma(1 + beta) - np.lgamma((1 + beta) / 2))) ** (1 / beta)
        s = np.random.standard_normal(self.dim) * sigma
        step = s / np.abs(np.random.standard_normal(self.dim)) ** (1 / beta)
        levy = 0.01 * step * np.random.standard_normal(self.dim)
        return levy

    def __call__(self, func):
        pop_size, F, CR, bounds = 10, 0.5, 0.9, (-5.0, 5.0)
        best_solution, best_fitness = np.random.uniform(*bounds, self.dim), func(np.random.uniform(*bounds, self.dim))

        for _ in range(self.budget):
            new_population = []
            for _ in range(pop_size):
                a, b, c = np.random.choice(pop_size, 3, replace=False)
                mutant = best_solution + F * (best_solution - new_population[a]) + self.levy_flight()
                trial = np.clip(mutant, *bounds)
                
                crossover_mask = np.random.rand(self.dim) < CR
                new_vector = np.where(crossover_mask, trial, best_solution)
                
                new_fitness = func(new_vector)
                if new_fitness < best_fitness:
                    best_solution, best_fitness = new_vector, new_fitness
                    
                new_population.append(new_vector)
            
        return best_solution