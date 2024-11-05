import numpy as np

class AdaptiveParticleSwarmFusion:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(30, dim * 5)
        self.inertia_weight_max = 0.85
        self.inertia_weight_min = 0.3
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.dynamic_topology_change_prob = 0.2
        self.eval_count = 0

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))

        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        self.eval_count += self.population_size

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]

        while self.eval_count < self.budget:
            phase = self.eval_count / self.budget
            w = self.inertia_weight_max - ((self.inertia_weight_max - self.inertia_weight_min) * np.power(phase, 2))  # Quadratic decay
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - particles) +
                          self.social_coeff * r2 * (global_best_position - particles))
            particles += velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            scores = np.array([func(p) for p in particles])
            self.eval_count += self.population_size

            better_mask = scores < personal_best_scores
            personal_best_positions[better_mask] = particles[better_mask]
            personal_best_scores[better_mask] = scores[better_mask]

            global_best_index = np.argmin(personal_best_scores)
            if personal_best_scores[global_best_index] < global_best_score:
                global_best_score = personal_best_scores[global_best_index]
                global_best_position = personal_best_positions[global_best_index]

            if np.random.rand() < self.dynamic_topology_change_prob:
                neighbors = np.random.randint(0, self.population_size, self.population_size)
                disturbances = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
                velocities += disturbances * (particles[neighbors] - particles)

            for i in range(len(particles)):
                if self.eval_count >= self.budget:
                    break
                if np.random.rand() < 0.18:  # Slightly increased probability for random walk
                    walk_step = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    walk_score = func(walk_step)
                    self.eval_count += 1
                    if walk_score < personal_best_scores[i]:
                        personal_best_positions[i] = walk_step
                        personal_best_scores[i] = walk_score
                        if walk_score < global_best_score:
                            global_best_score = walk_score
                            global_best_position = walk_step

        return global_best_position