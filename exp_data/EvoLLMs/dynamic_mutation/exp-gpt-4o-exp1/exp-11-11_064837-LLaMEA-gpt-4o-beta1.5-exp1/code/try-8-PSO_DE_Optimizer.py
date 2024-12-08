import numpy as np

class PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.f = 0.5  # DE scaling factor, now adaptive
        self.cr = 0.9  # DE crossover probability, now adaptive
        self.inertia_weight = 0.7
        self.cognitive_component = 1.5
        self.social_component = 1.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        np.random.seed(42)

    def __call__(self, func):
        # Initialize particles
        particle_positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        particle_velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(particle_positions)
        personal_best_scores = np.full(self.pop_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            # Evaluate current particle positions
            scores = np.array([func(x) for x in particle_positions])
            evaluations += self.pop_size

            # Update personal and global bests
            better_mask = scores < personal_best_scores
            personal_best_positions[better_mask] = particle_positions[better_mask]
            personal_best_scores[better_mask] = scores[better_mask]

            if np.min(scores) < global_best_score:
                global_best_score = np.min(scores)
                global_best_position = particle_positions[np.argmin(scores)]

            # Adaptive parameters
            self.f = 0.4 + 0.1 * np.random.rand()  # line changed
            self.cr = 0.8 + 0.2 * np.random.rand()  # line changed

            # PSO update
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_component * r1 * (personal_best_positions[i] - particle_positions[i])
                social_velocity = self.social_component * r2 * (global_best_position - particle_positions[i])
                particle_velocities[i] = (self.inertia_weight * particle_velocities[i] +
                                          cognitive_velocity + social_velocity)
                particle_positions[i] += particle_velocities[i]
                particle_positions[i] = np.clip(particle_positions[i], self.lower_bound, self.upper_bound)

            # DE-like mutation and crossover
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = particle_positions[a] + self.f * (particle_positions[b] - particle_positions[c])  # line changed
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, particle_positions[i])

                trial_score = func(trial)  # line changed
                if trial_score < scores[i]:  # line changed
                    particle_positions[i] = trial
                    scores[i] = trial_score  # line changed
                    evaluations += 1

        return global_best_position