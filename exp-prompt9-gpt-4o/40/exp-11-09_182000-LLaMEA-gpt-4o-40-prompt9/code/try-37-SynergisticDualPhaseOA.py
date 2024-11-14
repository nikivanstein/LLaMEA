import numpy as np

class SynergisticDualPhaseOA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 60  # Increased population size for diversity
        self.w = 0.5  # More stable inertia weight for phase 1
        self.cr1 = 0.9  # Higher crossover probability for early phase
        self.cr2 = 0.7  # Lower crossover probability for later phase
        self.f1 = 0.4  # Smaller mutation factor initially
        self.f2 = 0.8  # Larger mutation factor for late exploration
        self.pm = 0.3  # Enhanced mutation probability for exploration
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
            phase_switch = self.budget // 3
            for i in range(self.pop_size):
                # Dual-Phase Mutation Strategy
                if evaluations < phase_switch:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x0, x1, x2 = population[indices]
                    mutant = x0 + self.f1 * (x1 - x2)
                    trial = np.where(np.random.rand(self.dim) < self.cr1, mutant, population[i])
                else:
                    mutant = personal_best[i] + self.f2 * (global_best - personal_best[i])
                    trial = np.where(np.random.rand(self.dim) < self.cr2, mutant, population[i])
                    
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
            
            # Dynamic Learning Rate PSO Update
            self.w = 0.6 + 0.3 * (1 - evaluations / self.budget)  # Decreasing inertia
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