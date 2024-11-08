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
                mutant = self.population[mutant_indices]
                trial = target + self.F * (mutant[0] - mutant[1])
                mask = np.random.rand(self.dim) < self.CR
                trial[mask] = target[mask]
                
                new_fitness = func(trial)
                if new_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = new_fitness
                    
                target = trial if new_fitness < func(target) else trial if np.exp((func(target) - new_fitness) / self.T) > np.random.rand() else target
                new_population.append(target)
                
            self.population = np.array(new_population)
            self.T = max(self.alpha * self.T, 0.1)
            
        return best_solution