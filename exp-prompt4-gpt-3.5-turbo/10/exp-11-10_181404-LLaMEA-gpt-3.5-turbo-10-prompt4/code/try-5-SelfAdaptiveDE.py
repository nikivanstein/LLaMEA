import numpy as np

class SelfAdaptiveDE:
    def __init__(self, budget, dim, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR

    def __call__(self, func):
        pop_size = 10 * self.dim
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        
        F_lower, F_upper = 0.2, 0.8  # Define lower and upper bounds for F
        CR_lower, CR_upper = 0.2, 1.0  # Define lower and upper bounds for CR
        
        for _ in range(self.budget - pop_size):
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Self-adaptive mutation strategy for F and CR
                F = np.clip(np.random.normal(self.F, 0.1), F_lower, F_upper)
                CR = np.clip(np.random.normal(self.CR, 0.1), CR_lower, CR_upper)
                
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f = func(trial)
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial
                    if f < fitness[best_idx]:
                        best_idx = i
                        best = trial
        
        return best