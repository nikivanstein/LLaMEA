import numpy as np

class EnhancedPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.F = 0.8
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.elitism_rate = 0.1  # Proportion of elite individuals retained

    def __call__(self, func):
        # Initialize swarm
        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = swarm.copy()
        personal_best_values = np.array([func(ind) for ind in swarm])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]
        
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            # Adaptive inertia weight
            self.w = 0.9 - (0.5 * evaluations / self.budget)
            
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - swarm[i]) +
                                 self.c2 * r2 * (global_best - swarm[i]))
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)
                
                f_val = func(swarm[i])
                evaluations += 1
                
                if f_val < personal_best_values[i]:
                    personal_best_values[i] = f_val
                    personal_best[i] = swarm[i].copy()
                    
                    if f_val < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]

                if evaluations >= self.budget:
                    break
            
            # Apply elitism
            elite_count = int(self.elitism_rate * self.pop_size)
            elite_indices = personal_best_values.argsort()[:elite_count]
            elites = personal_best[elite_indices]

            # Differential Evolution (DE) mutation and crossover
            for i in range(self.pop_size):
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                x1, x2, x3 = swarm[indices]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, swarm[i])
                
                f_trial = func(trial)
                evaluations += 1
                
                if f_trial < personal_best_values[i]:
                    personal_best_values[i] = f_trial
                    personal_best[i] = trial.copy()
                    
                    if f_trial < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]

                if evaluations >= self.budget:
                    break

            # Reinforce elite individuals back into the population
            if evaluations < self.budget:
                for j in range(elite_count):
                    idx = np.random.randint(self.pop_size)
                    swarm[idx] = elites[j]
                    personal_best_values[idx] = func(swarm[idx])
                    evaluations += 1
                
                    if personal_best_values[idx] < personal_best_values[global_best_idx]:
                        global_best_idx = idx
                        global_best = personal_best[global_best_idx]

                    if evaluations >= self.budget:
                        break
        
        return global_best