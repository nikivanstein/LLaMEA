import numpy as np

class QuantumCooperativeOptimizerEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.max_iter = budget // self.pop_size
        self.F_base = 0.5  # Adjusted differential weight
        self.CR_base = 0.9  # Slightly increased crossover probability
        self.w_base = 0.3  # Adjusted inertia weight for PSO
        self.c1 = 2.0  # Further increased personal attraction coefficient
        self.c2 = 1.5  # Slightly decreased global attraction coefficient
        self.F_decay = 0.9  # Increased decay factor for differential weight
        self.CR_growth = 1.02  # Reduced growth factor for crossover probability

    def __call__(self, func):
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocity = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))
        
        fitness = np.apply_along_axis(func, 1, pop)
        personal_best = pop.copy()
        personal_best_fitness = fitness.copy()
        global_best = pop[np.argmin(fitness)]
        
        eval_count = self.pop_size
        
        for t in range(self.max_iter):
            F = self.F_base * (self.F_decay ** t)
            CR = min(self.CR_base * (self.CR_growth ** t), 1.0)
            w = self.w_base * (0.5 + 0.5 * np.cos((t / self.max_iter) * np.pi))

            diversity = np.std(pop, axis=0)
            diversity_factor = np.clip(1.0 + 0.1 * (diversity.mean() / (self.upper_bound - self.lower_bound)), 0.5, 1.5)

            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 5, replace=False)
                x0, x1, x2, x3, x4 = pop[indices]
                mutant = x0 + F * (x1 - x2) + F * (x3 - x4) * diversity_factor
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)
                eval_count += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = trial
                        personal_best_fitness[i] = trial_fitness
                        if trial_fitness < func(global_best):
                            global_best = trial

            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocity = (w * velocity + 
                        self.c1 * r1 * (personal_best - pop) + 
                        self.c2 * r2 * (global_best - pop))
            pop = np.clip(pop + velocity, self.lower_bound, self.upper_bound)
            
            fitness = np.apply_along_axis(func, 1, pop)
            eval_count += self.pop_size
            
            better_mask = fitness < personal_best_fitness
            personal_best[better_mask] = pop[better_mask]
            personal_best_fitness[better_mask] = fitness[better_mask]
            if np.min(fitness) < func(global_best):
                global_best = pop[np.argmin(fitness)]

            if eval_count >= self.budget:
                break

        return global_best