import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.F = 0.5
        self.CR = 0.9

    def __call__(self, func):
        # Initialize particles and velocities
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.apply_along_axis(func, 1, particles)
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2, self.population_size, self.dim)

            # PSO Update
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - particles) +
                          self.c2 * r2 * (global_best_position - particles))
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)

            # Evaluate new positions and update personal bests
            new_scores = np.apply_along_axis(func, 1, particles)
            evaluations += self.population_size
            better_mask = new_scores < personal_best_scores
            personal_best_scores = np.where(better_mask, new_scores, personal_best_scores)
            personal_best_positions = np.where(better_mask[:, None], particles, personal_best_positions)

            # DE Mutation and Crossover
            for i in range(self.population_size):
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant = np.clip(particles[indices[0]] + self.F * (particles[indices[1]] - particles[indices[2]]), self.lower_bound, self.upper_bound)
                jrand = np.random.randint(0, self.dim)
                crossover_mask = np.random.rand(self.dim) < self.CR
                crossover_mask[jrand] = True  # Ensure at least one dimension is taken
                trial = np.where(crossover_mask, mutant, particles[i])

                trial_score = func(trial)
                evaluations += 1
                if trial_score < new_scores[i]:
                    new_scores[i] = trial_score
                    particles[i] = trial

            # Update Global Best
            current_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[current_best_idx] < global_best_score:
                global_best_score = personal_best_scores[current_best_idx]
                global_best_position = personal_best_positions[current_best_idx]

            if evaluations >= self.budget:
                break

        return global_best_position, global_best_score