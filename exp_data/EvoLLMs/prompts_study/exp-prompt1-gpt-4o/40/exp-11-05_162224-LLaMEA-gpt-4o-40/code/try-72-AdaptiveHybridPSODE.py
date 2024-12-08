import numpy as np

class AdaptiveHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.c1_initial = 1.5  # Initial PSO cognitive coefficient
        self.c2_initial = 1.5  # Initial PSO social coefficient
        self.w_initial = 0.7  # Initial inertia weight
        self.f = 0.8  # DE scaling factor
        self.cr_initial = 0.9  # Initial DE crossover probability

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_scores = np.array([func(ind) for ind in pop])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        eval_count = self.pop_size

        while eval_count < self.budget:
            # Dynamic adjustment
            progress = eval_count / self.budget
            w = self.w_initial * (1 - progress)
            c1 = self.c1_initial * (1 + progress)
            c2 = self.c2_initial * (1 + progress)
            cr = self.cr_initial * (1 - progress)

            # Neighborhood learning
            for i in range(self.pop_size):
                neighbors = np.random.choice(self.pop_size, 3, replace=False)
                r1, r2 = np.random.rand(2)
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best_positions[neighbors[0]] - pop[i]) +
                                 c2 * r2 * (global_best_position - pop[i]))
                pop[i] += velocities[i]
                pop[i] = np.clip(pop[i], self.lower_bound, self.upper_bound)

            scores = np.array([func(ind) for ind in pop])
            eval_count += self.pop_size

            better_mask = scores < personal_best_scores
            personal_best_scores = np.where(better_mask, scores, personal_best_scores)
            personal_best_positions = np.where(better_mask[:, np.newaxis], pop, personal_best_positions)

            current_global_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[current_global_best_idx] < global_best_score:
                global_best_score = personal_best_scores[current_global_best_idx]
                global_best_position = personal_best_positions[current_global_best_idx]

            # Differential Evolution Update
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = personal_best_positions[a] + self.f * (personal_best_positions[b] - personal_best_positions[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                trial = np.where(np.random.rand(self.dim) < cr, mutant, pop[i])
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

        return global_best_position, global_best_score