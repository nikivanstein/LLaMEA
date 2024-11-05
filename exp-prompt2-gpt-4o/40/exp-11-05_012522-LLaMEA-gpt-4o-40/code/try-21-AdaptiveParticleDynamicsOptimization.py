import numpy as np

class AdaptiveParticleDynamicsOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 30
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = 0.7
        self.CR = 0.8
        self.diversity_threshold = 0.2

    def __call__(self, func):
        np.random.seed(42)
        pos = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        vel = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = pos.copy()
        personal_best_value = np.array([func(ind) for ind in personal_best])
        global_best_index = np.argmin(personal_best_value)
        global_best = personal_best[global_best_index]
        global_best_value = personal_best_value[global_best_index]

        evaluations = self.pop_size

        while evaluations < self.budget:
            w = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            vel = w * vel + self.c1 * r1 * (personal_best - pos) + self.c2 * r2 * (global_best - pos)
            pos = np.clip(pos + vel, self.lower_bound, self.upper_bound)

            fitness = np.array([func(ind) for ind in pos])
            evaluations += self.pop_size

            better_mask = fitness < personal_best_value
            personal_best[better_mask] = pos[better_mask]
            personal_best_value[better_mask] = fitness[better_mask]
            
            current_global_best_index = np.argmin(personal_best_value)
            if personal_best_value[current_global_best_index] < global_best_value:
                global_best = personal_best[current_global_best_index]
                global_best_value = personal_best_value[current_global_best_index]

            diversity = np.std(fitness)
            if diversity < self.diversity_threshold:
                for i in range(self.pop_size):
                    if evaluations >= self.budget:
                        break
                    idxs = np.random.choice(self.pop_size, 3, replace=False)
                    mutant = np.clip(personal_best[idxs[0]] + self.F * (personal_best[idxs[1]] - personal_best[idxs[2]]), self.lower_bound, self.upper_bound)

                    cross_points = np.random.rand(self.dim) < self.CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, pos[i])

                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < fitness[i]:
                        pos[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < global_best_value:
                            global_best = trial
                            global_best_value = trial_fitness

        return global_best, global_best_value