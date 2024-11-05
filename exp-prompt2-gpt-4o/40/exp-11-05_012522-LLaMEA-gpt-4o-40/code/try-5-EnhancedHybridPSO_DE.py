import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_pop_size = 20
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 2.0  # increased cognitive coefficient
        self.c2 = 2.0  # increased social coefficient
        self.F = 0.5 + np.random.rand() * 0.3  # adaptive F in range [0.5, 0.8]
        self.CR = 0.9  # crossover probability
        self.dynamic_pop_size = True

    def __call__(self, func):
        np.random.seed(42)
        pop_size = self.initial_pop_size
        pos = np.random.uniform(self.lower_bound, self.upper_bound, (pop_size, self.dim))
        vel = np.random.uniform(-1, 1, (pop_size, self.dim))
        personal_best = pos.copy()
        personal_best_value = np.array([func(ind) for ind in personal_best])
        global_best_index = np.argmin(personal_best_value)
        global_best = personal_best[global_best_index]
        global_best_value = personal_best_value[global_best_index]

        evaluations = pop_size
        
        while evaluations < self.budget:
            # Adjust inertia weight dynamically
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)

            # PSO step
            r1 = np.random.rand(pop_size, self.dim)
            r2 = np.random.rand(pop_size, self.dim)
            vel = w * vel + self.c1 * r1 * (personal_best - pos) + self.c2 * r2 * (global_best - pos)
            pos = np.clip(pos + vel, self.lower_bound, self.upper_bound)

            # Evaluate the candidate solutions
            fitness = np.array([func(ind) for ind in pos])
            evaluations += pop_size

            # Update personal and global bests
            better_mask = fitness < personal_best_value
            personal_best[better_mask] = pos[better_mask]
            personal_best_value[better_mask] = fitness[better_mask]
            
            current_global_best_index = np.argmin(personal_best_value)
            current_global_best_value = personal_best_value[current_global_best_index]
            if current_global_best_value < global_best_value:
                global_best = personal_best[current_global_best_index]
                global_best_value = current_global_best_value

            # Adjust population size dynamically
            if self.dynamic_pop_size and evaluations > self.budget // 2:
                pop_size = max(5, int(self.initial_pop_size * (1 - evaluations / self.budget)))

            # DE step
            for i in range(pop_size):
                if evaluations >= self.budget:
                    break
                # Mutation
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pos[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pos[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    pos[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < global_best_value:
                        global_best = trial
                        global_best_value = trial_fitness

        return global_best, global_best_value