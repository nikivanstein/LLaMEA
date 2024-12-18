import numpy as np

class ADERLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.best_solution = None
        self.best_score = float('inf')
        
    def __call__(self, func):
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        self.best_solution = population[np.argmin(scores)]
        self.best_score = np.min(scores)
        evaluations = population_size
        
        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                self.F = 0.3 + 0.5 * (np.random.rand() ** 2)
                mutant = np.clip(a + self.F * (b - c) + 0.3 * (self.best_solution - a), self.lower_bound, self.upper_bound)
                
                self.CR = 0.8 + 0.2 * np.random.rand()
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                
                if np.random.rand() < (0.15 + 0.1 * (evaluations / self.budget)):
                    trial = trial + np.random.normal(0, 0.15, self.dim)
                    trial = np.clip(trial, self.lower_bound, self.upper_bound)
                
                score = func(trial)
                evaluations += 1
                
                if score < scores[i]:
                    population[i] = trial
                    scores[i] = score
                    
                    if score < self.best_score:
                        self.best_solution = trial
                        self.best_score = score
            
            if evaluations % 100 == 0:
                population_size = max(4, population_size - 1)
                population = population[:population_size]
                scores = scores[:population_size]
        
        return self.best_solution