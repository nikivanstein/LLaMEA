import numpy as np

class EnhancedAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_particles = min(70, 12 * dim)
        self.min_particles = 30  # Introduced dynamic particle adjustment
        self.c1 = 1.5  # Slightly adjusted cognitive component for better individual learning
        self.c2 = 2.0  # Adjusted social component to maintain global influence
        self.w = 0.8  # Reduced inertia weight for quicker convergence
        self.decay_rate = 0.92  # Slightly faster decay for adaptive balance
        self.velocity_limit = (self.upper_bound - self.lower_bound) * 0.15  # More aggressive velocity limit

    def __call__(self, func):
        np.random.seed(42)
        num_particles = self.initial_particles
        pos = np.random.uniform(self.lower_bound, self.upper_bound, (num_particles, self.dim))
        vel = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = np.copy(pos)
        personal_best_scores = np.array([func(p) for p in pos])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = num_particles

        while evaluations < self.budget:
            for i in range(num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                vel[i] = (self.w * vel[i] +
                          self.c1 * r1 * (personal_best_positions[i] - pos[i]) +
                          self.c2 * r2 * (global_best_position - pos[i]))

                vel[i] = np.clip(vel[i], -self.velocity_limit, self.velocity_limit)
                pos[i] += vel[i]
                pos[i] = np.clip(pos[i], self.lower_bound, self.upper_bound)

                score = func(pos[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = pos[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = pos[i]

            self.w *= self.decay_rate  # Decay inertia weight

            # Adjust number of particles dynamically
            if evaluations < self.budget // 2:
                num_particles = max(self.min_particles, int(num_particles * 0.9))
                if num_particles < len(pos):
                    pos = pos[:num_particles]
                    vel = vel[:num_particles]
                    personal_best_positions = personal_best_positions[:num_particles]
                    personal_best_scores = personal_best_scores[:num_particles]

        return global_best_position, global_best_score