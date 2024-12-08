import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = min(50, max(10, budget // (8 * dim)))
        self.f = 0.5
        self.cr = 0.9
        self.w = 0.6  # Fixed initial inertia weight
        self.c1 = np.random.uniform(1.5, 2.5)
        self.c2 = np.random.uniform(1.0, 2.0)
        self.v_max = (self.upper_bound - self.lower_bound) / 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.pop_size

        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        global_best_idx = np.argmin(fitness)
        global_best = population[global_best_idx].copy()
        
        while budget_used < self.budget:
            # Differential Evolution with dynamic mutation
            for i in range(self.pop_size):
                if budget_used >= self.budget:
                    break
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[indices]
                f_dynamic = np.random.uniform(0.4, 0.9)
                mutant = np.clip(a + f_dynamic * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = trial
                        personal_best_fitness[i] = trial_fitness
                        if trial_fitness < personal_best_fitness[global_best_idx]:
                            global_best = trial
                            global_best_idx = i

            # Particle Swarm Optimization with adaptive inertia weight and local search
            self.w = 0.9 - 0.5 * (budget_used / self.budget)  # Adaptive inertia weight
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = np.clip(
                self.w * velocities 
                + self.c1 * r1 * (personal_best - population) 
                + self.c2 * r2 * (global_best - population),
                -self.v_max, self.v_max
            )
            population = np.clip(population + velocities, self.lower_bound, self.upper_bound)
            for i in range(self.pop_size):
                if budget_used >= self.budget:
                    break
                new_fitness = func(population[i])
                budget_used += 1
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = new_fitness
                    if new_fitness < personal_best_fitness[global_best_idx]:
                        global_best = population[i]
                        global_best_idx = i

                # Local search strategy
                if np.random.rand() < 0.1 and budget_used < self.budget:  # Perform local search with low probability
                    local_ind = population[i] + np.random.normal(0, 0.1, self.dim)
                    local_ind = np.clip(local_ind, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_ind)
                    budget_used += 1
                    if local_fitness < fitness[i]:
                        fitness[i] = local_fitness
                        personal_best[i] = local_ind
                        personal_best_fitness[i] = local_fitness
                        if local_fitness < personal_best_fitness[global_best_idx]:
                            global_best = local_ind
                            global_best_idx = i

        return global_best