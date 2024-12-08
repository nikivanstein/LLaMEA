import numpy as np

class DELF:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def levy_flight(self, beta=1.5):
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        s = np.random.normal(0, sigma, self.dim)
        u, v = np.random.normal(0, 1, (2, self.dim))
        step = s / np.abs(u) ** (1 / beta)
        return 0.01 * step * v

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
                best_solution, best_fitness = (new_vector, new_fitness) if new_fitness < best_fitness else (best_solution, best_fitness)
                new_population.append(new_vector)
            
        return best_solution