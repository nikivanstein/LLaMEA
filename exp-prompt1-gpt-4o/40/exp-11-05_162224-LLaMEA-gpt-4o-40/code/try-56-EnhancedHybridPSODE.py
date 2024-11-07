import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20  # Starting population size
        self.c1_start = 1.5  # Initial PSO cognitive coefficient
        self.c2_start = 1.5  # Initial PSO social coefficient
        self.w_start = 0.9  # Initial inertia weight
        self.f_start = 0.8  # Initial DE scaling factor
        self.cr = 0.9  # DE crossover probability

    def __call__(self, func):
        np.random.seed(42)
        # Adaptive parameters
        c1 = self.c1_start
        c2 = self.c2_start
        w = self.w_start
        f = self.f_start

        # Initialize population
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_scores = np.array([func(ind) for ind in pop])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        eval_count = self.pop_size

        while eval_count < self.budget:
            # Adaptive coefficients update
            progress_ratio = eval_count / self.budget
            c1 = self.c1_start - progress_ratio * 0.5
            c2 = self.c2_start + progress_ratio * 0.5
            w = self.w_start - progress_ratio * 0.5
            f = self.f_start - progress_ratio * 0.3

            # Particle Swarm Optimization Update
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities +
                          c1 * r1 * (personal_best_positions - pop) +
                          c2 * r2 * (global_best_position - pop))
            pop += velocities
            pop = np.clip(pop, self.lower_bound, self.upper_bound)

            # Evaluate the population
            scores = np.array([func(ind) for ind in pop])
            eval_count += self.pop_size

            # Update personal bests
            better_mask = scores < personal_best_scores
            personal_best_scores = np.where(better_mask, scores, personal_best_scores)
            personal_best_positions = np.where(better_mask[:, np.newaxis], pop, personal_best_positions)

            # Update global best
            current_global_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[current_global_best_idx] < global_best_score:
                global_best_score = personal_best_scores[current_global_best_idx]
                global_best_position = personal_best_positions[current_global_best_idx]

            # Differential Evolution Update
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = personal_best_positions[a] + f * (personal_best_positions[b] - personal_best_positions[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, pop[i])
                trial_score = func(trial)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial

                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = trial

                if eval_count >= self.budget:
                    break

            # Dynamic population adjustment (e.g., reduce population size for exploitation)
            if eval_count % (self.budget // 10) == 0 and self.pop_size > 10:
                self.pop_size -= 1
                pop = pop[:self.pop_size]
                velocities = velocities[:self.pop_size]
                personal_best_positions = personal_best_positions[:self.pop_size]
                personal_best_scores = personal_best_scores[:self.pop_size]

        return global_best_position, global_best_score