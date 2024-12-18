import numpy as np

class ADERLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.best_solution = None
        self.best_score = float('inf')
        
    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        self.best_solution = population[np.argmin(scores)]
        self.best_score = np.min(scores)
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Dynamic mutation factor with diversity influence
                diversity = np.std(population, axis=0).mean()
                self.F = 0.3 + 0.4 * np.random.rand() * (1 - diversity / (self.upper_bound - self.lower_bound))
                # Slight increase in the bias towards the global best
                mutant = np.clip(a + self.F * (b - c) + 0.25 * (self.best_solution - a), self.lower_bound, self.upper_bound)
                
                # Dynamic crossover probability
                self.CR = 0.85 + 0.15 * np.random.rand()
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                
                # Stochastic ranking to adjust local search
                if np.random.rand() < (0.1 + 0.2 * (np.argsort(scores).tolist().index(i) / self.population_size)):  # Updated dynamic probability
                    top_mean = np.mean(population[np.argsort(scores)[:int(self.population_size * 0.2)]], axis=0)
                    trial = trial + np.random.normal(0, 0.1, self.dim) + np.random.normal(0, 0.05, self.dim) + 0.05 * (top_mean - trial)
                    trial = np.clip(trial, self.lower_bound, self.upper_bound)
                
                score = func(trial)
                evaluations += 1
                
                if score < scores[i]:
                    population[i] = trial
                    scores[i] = score
                    
                    if score < self.best_score:
                        self.best_solution = trial
                        self.best_score = score
        
        return self.best_solution