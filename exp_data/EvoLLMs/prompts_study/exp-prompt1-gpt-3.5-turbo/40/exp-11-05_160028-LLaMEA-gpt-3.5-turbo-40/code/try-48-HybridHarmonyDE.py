import numpy as np
from scipy.optimize import differential_evolution

class HybridHarmonyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget - len(population)):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            new_fitness = func(new_harmony)
            
            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                population[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        bounds = [(self.lower_bound, self.upper_bound)] * self.dim
        result = differential_evolution(func, bounds, init="latinhypercube", maxiter=self.budget)
        if result.fun < func(best_solution):
            best_solution = result.x
        
        return best_solution