import numpy as np

class Enhanced_DE_SA_Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.CR = 0.5
        self.F = 0.5
        self.T = 1.0
        self.alpha = 0.9
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        
    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            new_population = []
            for target in self.population:
                mutant_indices = np.random.choice(range(self.pop_size), 2, replace=False)
                mutant_diff = self.population[mutant_indices[0]] - self.population[mutant_indices[1]]
                trial = target + self.F * mutant_diff
                mask = np.random.rand(self.dim) < self.CR
                trial = np.where(mask, trial, target)
                
                new_fitness = func(trial)
                if new_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = new_fitness
                    
                target = np.where(new_fitness < func(target) or np.exp((func(target) - new_fitness) / self.T) > np.random.rand(), trial, target)
                new_population.append(target)
                
            self.population = np.array(new_population)
            self.T = max(self.alpha * self.T, 0.1)
            
        return best_solution