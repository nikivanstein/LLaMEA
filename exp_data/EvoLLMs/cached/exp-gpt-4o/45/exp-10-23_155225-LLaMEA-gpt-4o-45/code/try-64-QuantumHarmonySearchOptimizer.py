import numpy as np

class QuantumHarmonySearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.max_iter = budget // self.pop_size
        self.F_base = 0.5  # Adjusted differential weight
        self.CR_base = 0.85  # Adjusted crossover probability
        self.w_base = 0.5  # Adjusted inertia weight for PSO
        self.c1 = 1.5  # Reduced personal attraction coefficient
        self.c2 = 1.9  # Increased global attraction coefficient
        self.harmony_memory_consideration_rate = 0.9  # New parameter for harmony memory consideration
        self.pitch_adjustment_rate = 0.3  # New parameter for pitch adjustment
        self.evaluations = 0

    def __call__(self, func):
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        
        fitness = np.apply_along_axis(func, 1, pop)
        personal_best = pop.copy()
        personal_best_fitness = fitness.copy()
        global_best = pop[np.argmin(fitness)]
        
        eval_count = self.pop_size
        
        for t in range(self.max_iter):
            F = self.F_base
            CR = self.CR_base
            w = self.w_base
            
            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 4, replace=False)
                x0, x1, x2, x3 = pop[indices]
                mutant = x0 + F * (x1 - x2) + F * (x3 - x0)
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
                
                # Harmony Search operations
                if np.random.rand() < self.harmony_memory_consideration_rate:
                    new_solution = np.random.choice(pop, size=self.dim, replace=True).mean(axis=0)
                    if np.random.rand() < self.pitch_adjustment_rate:
                        new_solution += np.random.normal(0, 1, self.dim)
                    new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                    new_fitness = func(new_solution)
                    eval_count += 1
                    if new_fitness < fitness[i]:
                        pop[i] = new_solution
                        fitness[i] = new_fitness

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