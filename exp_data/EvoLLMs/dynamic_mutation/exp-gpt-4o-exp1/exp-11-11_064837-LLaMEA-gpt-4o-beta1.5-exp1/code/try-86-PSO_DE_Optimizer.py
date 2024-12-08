import numpy as np

class PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.f = 0.5  # DE scaling factor
        self.cr = 0.9  # DE crossover probability
        self.inertia_weight = 0.7
        self.cognitive_component = 1.5
        self.social_component = 1.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        np.random.seed(42)

    def levy_flights(self, size):
        beta = 1.5
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)))**(1/beta)
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, 1, size)
        step = u / np.power(np.abs(v), 1/beta)
        return step

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

            # PSO update
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                adaptive_cognitive = self.cognitive_component * (1 - evaluations / self.budget)
                cognitive_velocity = adaptive_cognitive * r1 * (personal_best_positions[i] - particle_positions[i])
                adaptive_social = self.social_component * (1 - evaluations / self.budget)
                social_velocity = adaptive_social * r2 * (global_best_position - particle_positions[i])
                adaptive_inertia_weight = self.inertia_weight * (1 - evaluations / self.budget)
                adaptive_inertia_weight = np.random.uniform(0.5, 1.0, self.dim) * adaptive_inertia_weight
                particle_velocities[i] = (adaptive_inertia_weight * particle_velocities[i] +
                                          cognitive_velocity +
                                          social_velocity)
                # Changed line: Clip particle velocities to control velocity boundaries
                particle_velocities[i] = np.clip(particle_velocities[i], -1, 1)
                particle_positions[i] += particle_velocities[i]
                particle_positions[i] = np.clip(particle_positions[i], self.lower_bound, self.upper_bound)

            # DE-like mutation and crossover
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                dynamic_perturbation = np.random.uniform(-0.2, 0.2, self.dim) * (1 - evaluations / self.budget)
                gaussian_mutation = np.random.normal(0, 0.1 * (1 - evaluations / self.budget), self.dim)
                # Changed line: Introduce dynamic DE scaling factor
                dynamic_f = self.f * np.random.uniform(0.5, 1.0)
                mutant = dynamic_f * (particle_positions[a] + particle_positions[b]) + dynamic_perturbation + gaussian_mutation
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Adaptive crossover probability
                adaptive_cr = self.cr * (1 - evaluations / self.budget)
                trial = np.where(np.random.rand(self.dim) < adaptive_cr, mutant, particle_positions[i])

                if func(trial) < scores[i]:
                    particle_positions[i] = trial
                    scores[i] = func(trial)
                    evaluations += 1

            # Changed line: Apply Levy flights to enhance exploration
            if evaluations % (self.budget // 10) == 0:
                levy_step = self.levy_flights(self.dim)
                particle_positions += levy_step
                particle_positions = np.clip(particle_positions, self.lower_bound, self.upper_bound)

        return global_best_position