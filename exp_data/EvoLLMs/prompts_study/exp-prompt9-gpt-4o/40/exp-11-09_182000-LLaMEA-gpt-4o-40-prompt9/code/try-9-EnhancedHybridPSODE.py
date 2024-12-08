import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.c1 = 1.5  # Adaptive cognitive parameter
        self.c2 = 1.5  # Adaptive social parameter
        self.w = 0.6  # Adaptive inertia weight
        self.mutation_factor = 0.8  # Differential Evolution mutation factor
        self.crossover_prob = 0.9  # Differential Evolution crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize swarm
        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))
        personal_best = swarm.copy()
        personal_best_values = np.array([func(ind) for ind in swarm])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]
        
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            # Adaptive PSO Update
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - swarm[i]) +
                                 self.c2 * r2 * (global_best - swarm[i]))
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)
                
                # Evaluate new position
                f_val = func(swarm[i])
                evaluations += 1
                
                # Update personal and global bests
                if f_val < personal_best_values[i]:
                    personal_best_values[i] = f_val
                    personal_best[i] = swarm[i].copy()
                    
                    if f_val < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]

                if evaluations >= self.budget:
                    break

            # Differential Evolution inspired update
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = swarm[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                trial = np.copy(swarm[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_prob:
                        trial[j] = mutant[j]
                f_trial = func(trial)
                evaluations += 1
                if f_trial < personal_best_values[i]:
                    personal_best_values[i] = f_trial
                    personal_best[i] = trial
                    if f_trial < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]

            # Adaptive parameter adjustment
            self.w = 0.4 + 0.3 * np.random.rand()
            self.c1 = 1.0 + np.random.rand()
            self.c2 = 1.0 + np.random.rand()
        
        return global_best