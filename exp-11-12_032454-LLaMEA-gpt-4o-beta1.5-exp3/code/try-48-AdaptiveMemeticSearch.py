import numpy as np

class AdaptiveMemeticSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic for population size
        self.local_search_ratio = 0.2
        self.learning_rate = 0.5
        
    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        
        while num_evaluations < self.budget:
            # Global exploration using differential mutation
            new_population = []
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = a + self.learning_rate * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)
                
                # Local search with a probability
                if np.random.rand() < self.local_search_ratio:
                    local_search_step = np.random.randn(self.dim) * 0.1
                    mutant = np.clip(mutant + local_search_step, self.lb, self.ub)
                
                offspring_fitness = func(mutant)
                num_evaluations += 1
                
                if offspring_fitness < fitness[i]:
                    new_population.append(mutant)
                    fitness[i] = offspring_fitness
                    if offspring_fitness < best_fitness:
                        best_individual = mutant
                        best_fitness = offspring_fitness
                else:
                    new_population.append(population[i])
            
            population = np.array(new_population)
        
        return best_individual, best_fitness