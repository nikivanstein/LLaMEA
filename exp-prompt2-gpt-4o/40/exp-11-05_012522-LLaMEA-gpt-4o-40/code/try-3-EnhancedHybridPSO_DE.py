import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.initial_w = 0.9  # initial inertia weight
        self.final_w = 0.4  # final inertia weight
        self.c1 = 1.5  # cognitive (personal) coefficient
        self.c2 = 1.5  # social (global) coefficient
        self.F = 0.8  # differential weight
        self.CR = 0.9  # crossover probability

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
            w = self.initial_w - (self.initial_w - self.final_w) * evaluations / self.budget  # adaptive inertia weight

            # Particle Swarm Optimization step
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            vel = w * vel + self.c1 * r1 * (personal_best - pos) + self.c2 * r2 * (global_best - pos)
            pos = np.clip(pos + vel, self.lower_bound, self.upper_bound)

            # Evaluate the candidate solutions
            fitness = np.array([func(ind) for ind in pos])
            evaluations += self.pop_size

            # Update personal and global bests
            better_mask = fitness < personal_best_value
            personal_best[better_mask] = pos[better_mask]
            personal_best_value[better_mask] = fitness[better_mask]
            
            current_global_best_index = np.argmin(personal_best_value)
            current_global_best_value = personal_best_value[current_global_best_index]
            if current_global_best_value < global_best_value:
                global_best = personal_best[current_global_best_index]
                global_best_value = current_global_best_value

            # Differential Evolution step
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                # Mutation with adaptive differential weight
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pos[np.random.choice(idxs, 3, replace=False)]
                adaptive_F = self.F + 0.2 * np.random.rand()  # slight randomization
                mutant = np.clip(a + adaptive_F * (b - c), self.lower_bound, self.upper_bound)

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