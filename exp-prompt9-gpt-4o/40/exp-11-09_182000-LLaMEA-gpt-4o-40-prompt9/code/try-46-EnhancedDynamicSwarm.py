import numpy as np

class EnhancedDynamicSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50  # Reduced population size for focused exploration
        self.w = 0.7  # Adjusted inertia weight for improved exploration
        self.cr = 0.85  # Slightly reduced crossover to maintain necessary diversity
        self.f = 0.6  # Increased mutation factor for broader search space exploration
        self.pm = 0.15  # Increased mutation probability for more frequent updates
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-0.3, 0.3, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]
        
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Enhanced Mutation Strategies
                indices = np.random.choice(self.pop_size, 5, replace=False)
                x0, x1, x2, x3, x4 = population[indices]
                mutant = x0 + self.f * (x1 - x2) + self.f * (x3 - x4)
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

                if evaluations >= self.budget:
                    break
            
            # Adaptive PSO Update with Learning Factor
            self.w = 0.4 + 0.3 * np.exp(-evaluations / (self.budget / 10))  # Exponential decay of inertia
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