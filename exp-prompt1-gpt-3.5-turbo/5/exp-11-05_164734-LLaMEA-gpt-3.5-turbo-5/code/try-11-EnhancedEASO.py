import numpy as np

class EnhancedEASO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        def diff_evolution(population, f=0.5, cr=0.9):
            mutant_population = np.zeros_like(population)
            for i in range(len(population)):
                candidates = [idx for idx in range(len(population)) if idx != i]
                a, b, c = population[np.random.choice(candidates, 3, replace=False)]
                mutant = np.clip(a + f * (b - c), -5.0, 5.0)
                j_rand = np.random.randint(self.dim)
                trial = [mutant[j] if np.random.rand() < cr or j == j_rand else population[i][j] for j in range(self.dim)]
                mutant_population[i] = trial
            return mutant_population
        
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
            new_population = diff_evolution(population)
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