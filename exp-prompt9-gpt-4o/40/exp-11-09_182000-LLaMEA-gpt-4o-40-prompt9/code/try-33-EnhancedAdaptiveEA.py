import numpy as np

class EnhancedAdaptiveEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.w = 0.5  # Adjusted for faster convergence
        self.cr = 0.9  # Higher crossover probability
        self.f = 0.5  # Reduced mutation factor for stability
        self.pm = 0.3  # Further increased mutation probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def levy_flight(self, mu=1.5, scale=0.1):
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1/mu)
        return step

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-0.3, 0.3, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]
        
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations < self.budget // 2:
                    indices = np.random.choice(self.pop_size, 2, replace=False)
                    x1, x2 = population[indices]
                    mutant = population[i] + self.f * (x1 - x2) + self.levy_flight()
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

                if evaluations >= self.budget:
                    break
            
            self.w = 0.3 + 0.4 * (1 - evaluations / self.budget)
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 r1 * (personal_best[i] - population[i]) +
                                 r2 * (global_best - population[i]))
                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                
                f_val = func(population[i])
                evaluations += 1
                
                if f_val < personal_best_values[i]:
                    personal_best_values[i] = f_val
                    personal_best[i] = population[i].copy()
                    if f_val < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]
        
        return global_best