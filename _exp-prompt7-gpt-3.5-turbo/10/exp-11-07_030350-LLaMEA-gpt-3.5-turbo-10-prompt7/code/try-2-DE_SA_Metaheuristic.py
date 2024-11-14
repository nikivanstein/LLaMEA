import numpy as np

class DE_SA_Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.CR = 0.5
        self.F = 0.5
        self.T_init = 1.0
        self.T_min = 0.1
        self.alpha = 0.9
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        
    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        T = self.T_init
        
        for _ in range(self.budget):
            new_population = []
            for i in range(self.pop_size):
                target = self.population[i]
                mutant = self.population[np.random.choice(np.delete(np.arange(self.pop_size), i, axis=0), 2, replace=False)]
                trial = target + self.F * (mutant[0] - mutant[1])
                mask = np.random.rand(self.dim) < self.CR
                trial[mask] = target[mask]
                
                new_fitness = func(trial)
                if new_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = new_fitness
                if new_fitness < func(target):
                    target = trial
                else:
                    if np.exp((func(target) - new_fitness) / T) > np.random.rand():
                        target = trial
                new_population.append(target)
                
            self.population = np.array(new_population)
            T = max(self.alpha * T, self.T_min)
            
        return best_solution