import numpy as np

class ImprovedEnhancedDynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        best_solution = population[np.argmin([func(individual) for individual in population])]
        F = 0.5
        CR = 0.9
        diversity = 0.1
        
        for _ in range(self.budget):
            mutant_pop = []
            for idx, target in enumerate(population):
                a, b, c = np.random.choice([i for i in range(self.budget) if i != idx], 3, replace=False)
                mutant = population[a] + F * (population[b] - population[c]) + np.random.standard_cauchy(self.dim) * diversity
                mutant = np.clip(mutant, -5.0, 5.0)
                
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, target)
                
                if func(trial) < func(target):
                    population[idx] = trial
                    if func(trial) < func(best_solution):
                        best_solution = np.copy(trial)
            
            F = max(0.1, F * 0.99)
            CR = max(0.1, CR * 0.99)
            diversity = max(0.01, diversity * 0.99)
        
        return best_solution