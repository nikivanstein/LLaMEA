import numpy as np

class HybridGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic for population size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.adaptive_factor = 0.1

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        while num_evaluations < self.budget:
            # Mutation and Crossover
            new_population = np.zeros_like(population)
            for i in range(self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = population[np.random.choice(candidates, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lb, self.ub)
                
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                
                trial = np.where(crossover_mask, mutant, population[i])
                new_fitness = func(trial)
                num_evaluations += 1
                
                if new_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = new_fitness
                else:
                    new_population[i] = population[i]
                
                if num_evaluations >= self.budget:
                    break
            
            # Adaptive crossover adjustment
            self.crossover_rate = max(0.1, self.crossover_rate - self.adaptive_factor * (np.mean(fitness) / np.min(fitness)))
            population = new_population
        
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]