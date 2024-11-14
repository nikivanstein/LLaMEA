import numpy as np

class HADE_LS_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.55  # slightly increased scaling factor to enhance exploration
        self.CR = 0.85  # slightly reduced crossover probability to maintain diversity
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
            
            # Local search periodically with slight adjustment in conditions
            if evaluations < self.budget and evaluations % (self.population_size * 1.8) == 0:
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