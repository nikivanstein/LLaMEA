import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 50
        self.c1 = 2.05
        self.c2 = 2.05
        self.w = 0.7
        self.f = 0.5
        self.cr = 0.9
        
    def __call__(self, func):
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        pbest = pop.copy()
        pbest_fitness = np.apply_along_axis(func, 1, pbest)
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx]
        gbest_fitness = pbest_fitness[gbest_idx]
        
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            # Generate random coefficients once per iteration
            r1, r2 = np.random.rand(2, self.pop_size, self.dim)
            
            # Update velocities and positions in vectorized form
            velocities = (self.w * velocities +
                          self.c1 * r1 * (pbest - pop) +
                          self.c2 * r2 * (gbest - pop))
            pop = np.clip(pop + velocities, self.lb, self.ub)
            
            # Evaluate new population
            fitness = np.apply_along_axis(func, 1, pop)
            evaluations += self.pop_size
            
            # Update pbest and gbest
            improved = fitness < pbest_fitness
            np.copyto(pbest, pop, where=improved[:, np.newaxis])
            np.copyto(pbest_fitness, fitness, where=improved)
            
            min_idx = np.argmin(pbest_fitness)
            if pbest_fitness[min_idx] < gbest_fitness:
                gbest = pbest[min_idx]
                gbest_fitness = pbest_fitness[min_idx]
            
            # Differential Evolution step in a batch mode
            indices = np.array([np.random.choice(self.pop_size, 3, replace=False) for _ in range(self.pop_size)])
            a, b, c = indices[:, 0], indices[:, 1], indices[:, 2]
            mutants = np.clip(pbest[a] + self.f * (pbest[b] - pbest[c]), self.lb, self.ub)
            
            cross_points = np.random.rand(self.pop_size, self.dim) < self.cr
            trials = np.where(cross_points, mutants, pop)
            trial_fitness = np.apply_along_axis(func, 1, trials)
            evaluations += self.pop_size
            
            # Update based on trial fitness
            improved_trials = trial_fitness < pbest_fitness
            np.copyto(pbest, trials, where=improved_trials[:, np.newaxis])
            np.copyto(pbest_fitness, trial_fitness, where=improved_trials)

            # Update global best
            trial_min_idx = np.argmin(trial_fitness)
            if trial_fitness[trial_min_idx] < gbest_fitness:
                gbest = trials[trial_min_idx]
                gbest_fitness = trial_fitness[trial_min_idx]
        
        return gbest