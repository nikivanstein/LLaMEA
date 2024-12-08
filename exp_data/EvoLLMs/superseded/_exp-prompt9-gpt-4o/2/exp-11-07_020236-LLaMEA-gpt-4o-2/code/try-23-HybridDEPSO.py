import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.bounds = [-5.0, 5.0]
        self.vel_bounds = [-0.12, 0.12]  # Slightly wider velocity bounds
        self.F = 0.65  # Adjusted DE scaling factor for better exploration
        self.CR = 0.85  # Adjusted DE crossover probability for increased diversity
        self.w_max, self.w_min = 0.6, 0.35  # Narrower adaptive PSO inertia weight range
        self.c1, self.c2 = 1.5, 1.5  # Balanced PSO cognitive and social coefficients

    def __call__(self, func):
        np.random.seed(0)  # For reproducibility
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.random.uniform(self.vel_bounds[0], self.vel_bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        p_best = population.copy()
        p_best_fitness = fitness.copy()
        g_best_idx = np.argmin(fitness)
        g_best = population[g_best_idx].copy()
        g_best_fitness = fitness[g_best_idx]

        evaluations = self.pop_size

        while evaluations < self.budget:
            w = self.w_max - evaluations / self.budget * (self.w_max - self.w_min)  # Adaptive inertia
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
                    if trial_fitness < p_best_fitness[i]:
                        p_best[i], p_best_fitness[i] = trial, trial_fitness
                        if trial_fitness < g_best_fitness:
                            g_best, g_best_fitness = trial, trial_fitness

            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocities = (w * velocities +
                          self.c1 * r1 * (p_best - population) +
                          self.c2 * r2 * (g_best - population))
            velocities = np.clip(velocities, self.vel_bounds[0], self.vel_bounds[1])

            population = np.clip(population + velocities, self.bounds[0], self.bounds[1])
            new_fitness = np.array([func(ind) for ind in population])
            evaluations += self.pop_size

            for i in range(self.pop_size):
                if new_fitness[i] < p_best_fitness[i]:
                    p_best[i], p_best_fitness[i] = population[i].copy(), new_fitness[i]
                    if new_fitness[i] < g_best_fitness:
                        g_best, g_best_fitness = population[i].copy(), new_fitness[i]

        return g_best