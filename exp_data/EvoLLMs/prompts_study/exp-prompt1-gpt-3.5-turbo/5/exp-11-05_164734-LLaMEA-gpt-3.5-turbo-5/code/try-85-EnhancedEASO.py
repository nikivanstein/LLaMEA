import numpy as np

class EnhancedEASO(EASO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        def dynamic_mutate(x, sigma, fitness):
            return x + np.random.normal(0, sigma / (1 + np.std(fitness)), len(x))
        
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        sigma = 0.1
        temperature = 1.0
        cooling_rate = 0.9
        
        for _ in range(self.budget):
            new_population = np.array([dynamic_mutate(x, sigma, fitness) for x in population])
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