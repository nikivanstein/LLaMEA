import numpy as np

class ModifiedHybridEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 60  # Increased population size for better diversity
        self.alpha = 0.5  # Learning rate for adaptive step size
        self.beta = 0.3  # Secondary learning rate for diversity control
        self.pm = 0.25  # Mutation probability enhanced for exploration
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-0.2, 0.2, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]
        
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations < self.budget // 3:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x0, x1, x2 = population[indices]
                    mutant = x0 + self.alpha * (x1 - x2)
                elif evaluations < (2 * self.budget) // 3:
                    mutant = personal_best[i] + self.beta * (global_best - personal_best[i])
                else:
                    mutant = global_best + self.alpha * np.random.uniform(-1, 1, self.dim)
                    
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                trial = np.where(np.random.rand(self.dim) < 0.9, mutant, population[i])  # Improved crossover strategy
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
            
            # PSO Update with adaptive learning rate
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.alpha * velocities[i] +
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