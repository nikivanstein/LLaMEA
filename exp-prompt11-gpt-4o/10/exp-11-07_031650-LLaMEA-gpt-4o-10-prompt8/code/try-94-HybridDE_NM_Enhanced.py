import numpy as np
from scipy.optimize import minimize

class HybridDE_NM_Enhanced:
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
        self.f_values = np.full(self.population_size, np.inf)  # More efficient initialization

    def _differential_evolution(self, func):
        idxs = np.arange(self.population_size)
        for i in range(self.population_size):
            if self.func_evals >= self.budget:
                break

            a, b, c = self.pop[np.random.choice(idxs[idxs != i], 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)

            crossover = np.random.rand(self.dim) < self.CR
            crossover[np.random.choice(self.dim)] = True

            trial = np.where(crossover, mutant, self.pop[i])
            f_trial = func(trial)
            self.func_evals += 1

            if self.f_values[i] == np.inf:
                self.f_values[i] = func(self.pop[i])
                self.func_evals += 1

            if f_trial < self.f_values[i]:
                self.pop[i] = trial
                self.f_values[i] = f_trial
    
    def _nelder_mead(self, func, start_point):
        options = {'maxfev': self.budget - self.func_evals, 'disp': False}
        result = minimize(func, start_point, method='Nelder-Mead', options=options)
        self.func_evals += result.nfev
        return result.x
    
    def __call__(self, func):
        while self.func_evals < self.budget:
            self._differential_evolution(func)
        
        min_index = np.argmin(self.f_values)
        best_candidate = self.pop[min_index]
        best_solution = self._nelder_mead(func, best_candidate)
        
        return best_solution