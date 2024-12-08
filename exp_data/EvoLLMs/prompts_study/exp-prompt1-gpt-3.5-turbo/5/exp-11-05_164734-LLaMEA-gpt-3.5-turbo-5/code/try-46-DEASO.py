import numpy as np

class DEASO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        def mutate(x, sigma, population, f):
            a, b, c = population[np.random.choice(len(population), 3, replace=False)]
            return x + f * (a - x) + f * (b - c)
        
        def acceptance_probability(curr_fitness, new_fitness, temperature):
            if new_fitness < curr_fitness:
                return 1
            return np.exp((curr_fitness - new_fitness) / temperature)
        
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        sigma = 0.1
        temperature = 1.0
        cooling_rate = 0.9
        f = 0.5
        
        for _ in range(self.budget):
            new_population = np.array([mutate(x, sigma, population, f) for x in population])
            new_fitness = np.array([func(x) for x in new_population])
            
            for i in range(self.budget):
                if acceptance_probability(fitness[i], new_fitness[i], temperature) > np.random.rand():
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
            
            if np.min(fitness) < func(best_solution):
                best_idx = np.argmin(fitness)
                best_solution = population[best_idx]
            
            temperature *= cooling_rate
            sigma *= cooling_rate
        
        return best_solution