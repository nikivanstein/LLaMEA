import numpy as np

class EnhancedPSO_DE_MetaLearning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 30
        self.w = 0.5  # inertia weight
        self.c1 = 1.7  # cognitive coefficient
        self.c2 = 1.7  # social coefficient
        self.F = 0.7  # differential weight
        self.CR = 0.9  # crossover probability
        self.meta_learn_rate = 0.05  # rate to adjust hyperparameters
        self.diversity_factor = 0.1  # factor for diversity-based exploration

    def __call__(self, func):
        np.random.seed(42)
        pos = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        vel = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))
        personal_best = pos.copy()
        personal_best_value = np.array([func(ind) for ind in personal_best])
        global_best_index = np.argmin(personal_best_value)
        global_best = personal_best[global_best_index]
        global_best_value = personal_best_value[global_best_index]

        evaluations = self.pop_size
        
        while evaluations < self.budget:
            # Meta-learning adjustment
            improvement_rate = np.mean(personal_best_value - np.min(personal_best_value))
            if improvement_rate < 0.01:
                self.w += self.meta_learn_rate
                self.c1 -= self.meta_learn_rate
                self.c2 -= self.meta_learn_rate
            else:
                self.w -= self.meta_learn_rate
                self.c1 += self.meta_learn_rate
                self.c2 += self.meta_learn_rate

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

            # Differential Evolution step with diversity enhancement
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pos[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c) + self.diversity_factor * np.random.standard_normal(self.dim), self.lower_bound, self.upper_bound)

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