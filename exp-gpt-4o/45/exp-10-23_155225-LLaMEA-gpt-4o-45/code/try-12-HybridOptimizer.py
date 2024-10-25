import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.max_iter = budget // self.pop_size
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.w = 0.5  # Inertia weight for PSO
        self.c1 = 1.5  # Personal attraction coefficient
        self.c2 = 1.5  # Global attraction coefficient
        self.initial_temp = 1.0  # Initial temperature for SA

    def __call__(self, func):
        # Initialize population and velocity
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        
        # Evaluate initial population
        fitness = np.apply_along_axis(func, 1, pop)
        personal_best = pop.copy()
        personal_best_fitness = fitness.copy()
        global_best = pop[np.argmin(fitness)]
        
        eval_count = self.pop_size
        temperature = self.initial_temp

        for t in range(self.max_iter):
            # Differential Evolution step
            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 4, replace=False)
                x0, x1, x2, x3 = pop[indices]
                mutant = x0 + self.F * (x1 - x2) + self.F * (x3 - x0)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = trial
                        personal_best_fitness[i] = trial_fitness
                        if trial_fitness < func(global_best):
                            global_best = trial

            # Particle Swarm Optimization step
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocity = (self.w * velocity + 
                        self.c1 * r1 * (personal_best - pop) + 
                        self.c2 * r2 * (global_best - pop))
            pop = np.clip(pop + velocity, self.lower_bound, self.upper_bound)
            
            # Evaluate the new population
            fitness = np.apply_along_axis(func, 1, pop)
            eval_count += self.pop_size
            
            # Update personal and global bests
            better_mask = fitness < personal_best_fitness
            personal_best[better_mask] = pop[better_mask]
            personal_best_fitness[better_mask] = fitness[better_mask]
            if np.min(fitness) < func(global_best):
                global_best = pop[np.argmin(fitness)]

            # Update temperature
            temperature *= 0.95  # Cooling schedule

            if eval_count >= self.budget:
                break

        return global_best