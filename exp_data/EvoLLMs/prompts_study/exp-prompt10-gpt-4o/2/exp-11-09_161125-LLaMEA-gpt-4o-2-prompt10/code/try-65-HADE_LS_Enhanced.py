import numpy as np

class HADE_LS_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # scaling factor for mutation
        self.CR = 0.9  # crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def __call__(self, func):
        np.random.seed(42)  # for reproducibility
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        step_size = 0.1  # Adaptive step size for local search

        while evaluations < self.budget:
            self.adjust_population(fitness)
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                mutant = np.clip(population[a] + self.F * (population[b] - population[c]), self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < self.CR
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
            
            if evaluations % (self.population_size * 2) == 0:
                best = self.local_search(best, func, evaluations, step_size)
                step_size = max(step_size * 0.9, 0.01)  # Reduce step size adaptively
        
        return best
    
    def local_search(self, start, func, evaluations, step_size):
        current = start
        for _ in range(5):  # Reduce local search iterations
            neighbors = current + np.random.uniform(-step_size, step_size, self.dim)
            neighbors = np.clip(neighbors, self.lower_bound, self.upper_bound)
            f_neighbors = func(neighbors)
            evaluations += 1
            if f_neighbors < func(current):
                current = neighbors
            if evaluations >= self.budget:
                break
        return current
    
    def adjust_population(self, fitness):
        sorted_indices = np.argsort(fitness)
        top_individuals = int(0.2 * self.population_size)
        for idx in sorted_indices[:top_individuals]:
            self.F *= 0.98  # Slightly decrease F to favor exploration
            self.CR = min(self.CR + 0.01, 1.0)  # Slightly increase CR for more aggressive exploitation