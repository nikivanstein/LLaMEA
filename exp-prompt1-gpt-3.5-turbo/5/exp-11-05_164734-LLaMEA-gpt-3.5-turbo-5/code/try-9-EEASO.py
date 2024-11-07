import numpy as np

class EEASO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.elite_percentage = 0.1
    
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
        
        for _ in range(self.budget):
            new_population = np.array([mutate(x, sigma) for x in population])
            new_fitness = np.array([func(x) for x in new_population])
            
            elites_num = int(self.elite_percentage * self.budget)
            elite_indices = np.argsort(fitness)[:elites_num]
            
            for idx in elite_indices:
                if new_fitness[idx] < fitness[idx]:
                    population[idx] = new_population[idx]
                    fitness[idx] = new_fitness[idx]
            
            for i in range(self.budget):
                if i not in elite_indices and acceptance_probability(fitness[i], new_fitness[i], temperature) > np.random.rand():
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
            
            if np.min(fitness) < func(best_solution):
                best_idx = np.argmin(fitness)
                best_solution = population[best_idx]
            
            temperature *= cooling_rate
            sigma *= cooling_rate
        
        return best_solution