import numpy as np

class AdaptiveHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_init = 0.9
        self.w_final = 0.4
        self.F = 0.8
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0

    def chaotic_map(self, r=0.7):
        return r * (1 - r)

    def adaptive_weight(self):
        t = self.evaluations / self.budget
        return self.w_final + (self.w_init - self.w_final) * (1 - t)**2

    def __call__(self, func):
        # Initialize swarm
        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = swarm.copy()
        personal_best_values = np.array([func(ind) for ind in swarm])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]
        
        self.evaluations = self.pop_size
        
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                self.w = self.adaptive_weight()
                r1, r2 = np.random.rand(), np.random.rand()
                chaotic_factor = self.chaotic_map(np.random.rand())
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * chaotic_factor * (personal_best[i] - swarm[i]) +
                                 self.c2 * r2 * chaotic_factor * (global_best - swarm[i]))
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)
                
                # Evaluate new position
                f_val = func(swarm[i])
                self.evaluations += 1
                
                # Update personal and global bests
                if f_val < personal_best_values[i]:
                    personal_best_values[i] = f_val
                    personal_best[i] = swarm[i].copy()
                    
                    if f_val < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]

                if self.evaluations >= self.budget:
                    break
            
            # Differential Evolution (DE) mutation and crossover
            for i in range(self.pop_size):
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                x1, x2, x3 = swarm[indices]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, swarm[i])
                
                f_trial = func(trial)
                self.evaluations += 1
                
                if f_trial < personal_best_values[i]:
                    personal_best_values[i] = f_trial
                    personal_best[i] = trial.copy()
                    
                    if f_trial < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]

                if self.evaluations >= self.budget:
                    break
        
        return global_best