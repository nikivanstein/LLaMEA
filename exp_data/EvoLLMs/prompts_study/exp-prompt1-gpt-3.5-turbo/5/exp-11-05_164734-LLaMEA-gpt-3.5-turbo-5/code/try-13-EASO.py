import numpy as np

class EASO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        def mutate(x, sigma):
            return x + np.random.normal(0, sigma, len(x))
        
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
        
        adaptive_scale = 0.1  # Adaptive mutation scale
        
        for _ in range(self.budget):
            new_population = np.array([mutate(x, sigma) for x in population])
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
            sigma = max(adaptive_scale, sigma - adaptive_scale*self.budget/self.dim)  # Adaptive mutation scale update
        
        return best_solution