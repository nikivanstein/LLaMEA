import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 25
        self.w = 0.9  # initial inertia weight
        self.c1 = 2.0  # initial cognitive coefficient
        self.c2 = 1.5  # initial social coefficient
        self.F = 0.8  # differential weight
        self.CR = 0.9  # crossover probability
        self.local_search_prob = 0.15  # increased probability for local search

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
            # Adaptive learning rates
            self.w = 0.4 + 0.5 * np.random.rand()
            self.c1 = 1.5 + 0.5 * np.random.rand()
            self.c2 = 1.5 + 0.5 * np.random.rand()

            # Particle Swarm Optimization step with adaptive parameters
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            vel = self.w * vel + self.c1 * r1 * (personal_best - pos) + self.c2 * r2 * (global_best - pos)
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
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
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

            # Gradient-based local search
            if np.random.rand() < self.local_search_prob:
                for i in range(self.pop_size):
                    if evaluations >= self.budget:
                        break
                    grad = np.gradient(fitness)
                    local_candidate = pos[i] - 0.01 * grad[i]
                    local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_candidate)
                    evaluations += 1
                    if local_fitness < fitness[i]:
                        pos[i] = local_candidate
                        fitness[i] = local_fitness
                        if local_fitness < global_best_value:
                            global_best = local_candidate
                            global_best_value = local_fitness

        return global_best, global_best_value