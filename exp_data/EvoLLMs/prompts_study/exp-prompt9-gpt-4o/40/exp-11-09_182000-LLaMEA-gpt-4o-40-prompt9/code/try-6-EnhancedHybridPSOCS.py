import numpy as np

class EnhancedHybridPSOCS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.c1 = 1.5  # Tuned cognitive parameter
        self.c2 = 1.5  # Tuned social parameter
        self.w_start = 0.9  # Starting inertia weight
        self.w_end = 0.4    # Ending inertia weight
        self.pa = 0.25
        self.mutation_factor = 0.8  # Differential evolution mutation factor
        self.lower_bound = -5.0
        self.upper_bound = 5.0

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
            # Dynamic inertia computation
            w = self.w_end + (self.w_start - self.w_end) * ((self.budget - evaluations) / self.budget)
            
            # PSO Update with dynamic inertia
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (w * velocities[i] +
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

            # Differential Evolution (DE) inspired mutation
            for i in range(self.pop_size):
                if np.random.rand() < self.pa:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    a, b, c = swarm[indices[0]], swarm[indices[1]], swarm[indices[2]]
                    mutant_vector = a + self.mutation_factor * (b - c)
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                    
                    f_new = func(mutant_vector)
                    evaluations += 1
                    
                    if f_new < personal_best_values[i]:
                        personal_best_values[i] = f_new
                        personal_best[i] = mutant_vector.copy()
                        
                        if f_new < personal_best_values[global_best_idx]:
                            global_best_idx = i
                            global_best = personal_best[global_best_idx]

                if evaluations >= self.budget:
                    break
        
        return global_best