import numpy as np

class HADE_LS_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # initial scaling factor for mutation
        self.CR = 0.9  # initial crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def __call__(self, func):
        np.random.seed(42)  # for reproducibility
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best = population[best_idx]

        while evaluations < self.budget:
            population_std = np.std(population, axis=0)
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Adjust F and CR according to population diversity
                F_dynamic = self.F + 0.1 * np.mean(population_std)
                CR_dynamic = self.CR - 0.1 * np.mean(population_std)
                
                mutant = np.clip(population[a] + F_dynamic * (population[b] - population[c]), self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < CR_dynamic
                trial = np.where(crossover, mutant, population[i])
                
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_idx]:
                        best_idx = i
                        best = trial
                
                if evaluations >= self.budget:
                    break
            
            # Local search periodically
            if evaluations < self.budget and evaluations % (self.population_size * 2) == 0:
                best = self.local_search(best, func, evaluations)
        
        return best
    
    def local_search(self, start, func, evaluations):
        current = start
        step_size = 0.1
        for _ in range(10):  # limit local search iterations
            neighbors = current + np.random.uniform(-step_size, step_size, self.dim)
            neighbors = np.clip(neighbors, self.lower_bound, self.upper_bound)
            f_neighbors = func(neighbors)
            evaluations += 1
            if f_neighbors < func(current):
                current = neighbors
            if evaluations >= self.budget:
                break
        return current