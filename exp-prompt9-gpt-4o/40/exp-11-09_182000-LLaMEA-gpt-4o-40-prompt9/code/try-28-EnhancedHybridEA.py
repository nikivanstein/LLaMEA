import numpy as np

class EnhancedHybridEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 60  # Increased population size for diversity
        self.w = 0.5  # Adaptive inertia weight for balanced exploration
        self.cr = 0.9  # Higher crossover probability for diversity
        self.f = 0.5  # Reduced mutation factor for stability
        self.pm = 0.25  # Further increased mutation probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]
        
        evaluations = self.pop_size
        
        # Adaptive DE and PSO loop
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                
                # Adaptive DE Mutation
                if evaluations < 0.3 * self.budget:
                    indices = np.random.choice(self.pop_size, 4, replace=False)
                    x0, x1, x2, x3 = population[indices]
                    mutant = x0 + self.f * (x1 - x2) + self.f * (x3 - global_best)
                else:
                    r = np.random.rand()
                    if r < 0.7:  # More exploration in early stages
                        indices = np.random.choice(self.pop_size, 2, replace=False)
                        x0, x1 = population[indices]
                        mutant = x0 + self.f * (x1 - personal_best[i])
                    else:
                        mutant = personal_best[i] + self.f * (global_best - personal_best[i])
                
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, population[i])
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                
                f_trial = func(trial)
                evaluations += 1

                if f_trial < personal_best_values[i]:
                    personal_best_values[i] = f_trial
                    personal_best[i] = trial.copy()
                    
                    if f_trial < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]

            # PSO Update with adaptive inertia weight
            self.w = 0.5 + 0.2 * (1 - evaluations / self.budget)  # More inertia at start, less towards end
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 r1 * (personal_best[i] - population[i]) +
                                 r2 * (global_best - population[i]))
                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                
                # Evaluate new position
                f_val = func(population[i])
                evaluations += 1
                
                if f_val < personal_best_values[i]:
                    personal_best_values[i] = f_val
                    personal_best[i] = population[i].copy()
                    
                    if f_val < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]
        
        return global_best