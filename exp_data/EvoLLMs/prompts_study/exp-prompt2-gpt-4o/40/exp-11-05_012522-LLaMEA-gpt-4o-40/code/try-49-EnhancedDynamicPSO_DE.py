import numpy as np

class EnhancedDynamicPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = 0.8
        self.CR = 0.9
        self.adaptive_pop_size = self.pop_size

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
            # Adaptive inertia weight for velocity control
            self.w = 0.9 - 0.7 * (evaluations / self.budget)
            self.c1 = 1.5 - 1.0 * (evaluations / self.budget)
            self.c2 = 1.5 + 1.5 * (evaluations / self.budget)

            if evaluations % (self.pop_size * 5) == 0:
                if np.std(personal_best_value) < 0.1:
                    self.adaptive_pop_size = min(self.adaptive_pop_size + 5, 50)
                else:
                    self.adaptive_pop_size = max(self.adaptive_pop_size - 5, 10)
                pos = np.resize(pos, (self.adaptive_pop_size, self.dim))
                vel = np.resize(vel, (self.adaptive_pop_size, self.dim))

            r1 = np.random.rand(self.adaptive_pop_size, self.dim)
            r2 = np.random.rand(self.adaptive_pop_size, self.dim)
            vel = self.w * vel + self.c1 * r1 * (personal_best[:self.adaptive_pop_size] - pos) + self.c2 * r2 * (global_best - pos)
            pos = np.clip(pos + vel, self.lower_bound, self.upper_bound)

            fitness = np.array([func(ind) for ind in pos])
            evaluations += self.adaptive_pop_size

            better_mask = fitness < personal_best_value[:self.adaptive_pop_size]
            personal_best[:self.adaptive_pop_size][better_mask] = pos[better_mask]
            personal_best_value[:self.adaptive_pop_size][better_mask] = fitness[better_mask]
            
            current_global_best_index = np.argmin(personal_best_value[:self.adaptive_pop_size])
            current_global_best_value = personal_best_value[current_global_best_index]
            if current_global_best_value < global_best_value:
                global_best = personal_best[current_global_best_index]
                global_best_value = current_global_best_value

            for i in range(self.adaptive_pop_size):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.adaptive_pop_size) if idx != i]
                a, b, c = pos[np.random.choice(idxs, 3, replace=False)]
                mutant_1 = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                mutant_2 = np.clip(pos[i] + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial_1 = np.where(cross_points, mutant_1, pos[i])
                trial_2 = np.where(cross_points, mutant_2, pos[i])

                trial_1_fitness = func(trial_1)
                trial_2_fitness = func(trial_2)
                evaluations += 2
                if trial_1_fitness < fitness[i]:
                    pos[i] = trial_1
                    fitness[i] = trial_1_fitness
                    if trial_1_fitness < global_best_value:
                        global_best = trial_1
                        global_best_value = trial_1_fitness
                elif trial_2_fitness < fitness[i]:
                    pos[i] = trial_2
                    fitness[i] = trial_2_fitness
                    if trial_2_fitness < global_best_value:
                        global_best = trial_2
                        global_best_value = trial_2_fitness

        return global_best, global_best_value