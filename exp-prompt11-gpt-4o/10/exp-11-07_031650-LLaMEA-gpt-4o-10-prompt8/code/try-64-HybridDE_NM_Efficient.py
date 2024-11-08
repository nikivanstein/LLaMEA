import numpy as np
from scipy.optimize import minimize

class HybridDE_NM_Efficient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.lb = -5.0
        self.ub = 5.0
        self.pop = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.func_evals = 0
        self.f_values = np.array([float('inf')]*self.population_size)  # Track evaluations

    def _differential_evolution(self, func):
        for i in range(self.population_size):
            if self.func_evals >= self.budget:
                break

            idxs = np.delete(np.arange(self.population_size), i)  # Efficient index selection
            a, b, c = self.pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)
            
            crossover = np.random.rand(self.dim) < self.CR
            if not np.any(crossover):
                crossover[np.random.randint(0, self.dim)] = True
            
            trial = np.where(crossover, mutant, self.pop[i])
            f_trial = func(trial)
            self.func_evals += 1

            if self.f_values[i] == float('inf'):  # Evaluate only if not done before
                self.f_values[i] = func(self.pop[i])
                self.func_evals += 1

            if f_trial < self.f_values[i]:
                self.pop[i] = trial
                self.f_values[i] = f_trial
    
    def _nelder_mead(self, func, start_point):
        bounds = [(self.lb, self.ub)] * self.dim
        result = minimize(func, start_point, method='Nelder-Mead', bounds=bounds, options={'maxfev': self.budget - self.func_evals, 'disp': False})
        self.func_evals += result.nfev
        return result.x
    
    def __call__(self, func):
        while self.func_evals < self.budget:
            self._differential_evolution(func)
        
        min_index = np.argmin(self.f_values)  # Efficiently find the best candidate
        best_candidate = self.pop[min_index]
        best_solution = self._nelder_mead(func, best_candidate)
        
        return best_solution