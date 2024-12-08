import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 50  # Population size
        self.c1 = 2.05  # Cognitive factor
        self.c2 = 2.05  # Social factor
        self.w = 0.7    # Inertia weight
        self.f = 0.5    # DE mutation factor
        self.cr = 0.9   # DE crossover rate
        
    def __call__(self, func):
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        pbest = pop.copy()
        pbest_fitness = np.array([func(ind) for ind in pbest])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx]
        gbest_fitness = pbest_fitness[gbest_idx]
        
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            # Update velocities and positions (PSO)
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (pbest - pop) +
                          self.c2 * r2 * (gbest - pop))
            pop = pop + velocities
            pop = np.clip(pop, self.lb, self.ub)
            
            # Evaluate new population
            fitness = np.array([func(ind) for ind in pop])
            evaluations += self.pop_size
            
            # Update pbest and gbest
            for i in range(self.pop_size):
                if fitness[i] < pbest_fitness[i]:
                    pbest[i] = pop[i]
                    pbest_fitness[i] = fitness[i]
                    if pbest_fitness[i] < gbest_fitness:
                        gbest = pbest[i]
                        gbest_fitness = pbest_fitness[i]
            
            # Differential Evolution step
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(pbest[a] + self.f * (pbest[b] - pbest[c]), self.lb, self.ub)
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, pop[i])
                
                trial_fitness = func(trial)
                evaluations += 1
                
                if trial_fitness < pbest_fitness[i]:
                    pbest[i] = trial
                    pbest_fitness[i] = trial_fitness
                    if trial_fitness < gbest_fitness:
                        gbest = trial
                        gbest_fitness = trial_fitness
                
                if evaluations >= self.budget:
                    break
        
        return gbest