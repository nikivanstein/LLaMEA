import numpy as np

class Optimized_DE_SA_Metaheuristic:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.pop_size, self.CR, self.F = budget, dim, 10, 0.5, 0.5
        self.T_init, self.T_min, self.alpha = 1.0, 0.1, 0.9
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        
    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        T, budget = self.T_init, self.budget
        
        while budget > 0:
            new_population = []
            for i in range(self.pop_size):
                target, mutant = self.population[i], self.population[np.random.choice(np.delete(np.arange(self.pop_size), i, axis=0), 2, replace=False)]
                trial = target + self.F * (mutant[0] - mutant[1])
                mask = np.random.rand(self.dim) < self.CR
                trial[mask] = target[mask]
                
                new_fitness = func(trial)
                if new_fitness < best_fitness:
                    best_solution, best_fitness = trial, new_fitness
                if new_fitness < func(target):
                    target = trial
                elif np.exp((func(target) - new_fitness) / T) > np.random.rand():
                    target = trial
                new_population.append(target)
                
            self.population, T, budget = np.array(new_population), max(self.alpha * T, self.T_min), budget - 1
            
        return best_solution