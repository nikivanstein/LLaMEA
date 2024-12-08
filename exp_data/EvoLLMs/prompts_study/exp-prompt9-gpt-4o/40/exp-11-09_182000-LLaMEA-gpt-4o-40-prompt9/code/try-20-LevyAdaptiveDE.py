import numpy as np

class LevyAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.cr = 0.9  # Crossover probability for DE
        self.f = 0.8   # Differential weight for DE
        self.pm = 0.1  # Probability of mutation
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.alpha = 1.5  # Parameter for Levy flight

    def levy_flight(self):
        sigma = (np.math.gamma(1 + self.alpha) * np.sin(np.pi * self.alpha / 2) /
                 (np.math.gamma((1 + self.alpha) / 2) * self.alpha *
                 2**((self.alpha - 1) / 2)))**(1 / self.alpha)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / abs(v)**(1 / self.alpha)
        return step

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]
        
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            # DE Update
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, population[i])
                
                # Evaluate trial
                f_val = func(trial)
                evaluations += 1
                
                # Update personal and global bests
                if f_val < personal_best_values[i]:
                    personal_best_values[i] = f_val
                    personal_best[i] = trial.copy()
                    
                    if f_val < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]

                if evaluations >= self.budget:
                    break

            # Levy flight for exploration
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                step = self.levy_flight()
                new_pos = population[i] + step
                new_pos = np.clip(new_pos, self.lower_bound, self.upper_bound)
                
                f_new_pos = func(new_pos)
                evaluations += 1
                
                if f_new_pos < personal_best_values[i]:
                    personal_best_values[i] = f_new_pos
                    personal_best[i] = new_pos.copy()
                    
                    if f_new_pos < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]
        
        return global_best