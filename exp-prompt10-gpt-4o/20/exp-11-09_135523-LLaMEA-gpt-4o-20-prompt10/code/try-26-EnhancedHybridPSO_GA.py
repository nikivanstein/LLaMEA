import numpy as np

class EnhancedHybridPSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.init_particles = 60  # Increased initial swarm size for better exploration
        self.final_particles = 20  # Reduced final swarm size for focused exploitation
        self.w = 0.4  # Lower inertia for faster convergence
        self.c1 = 2.0  # Increased cognitive component for more personal best emphasis
        self.c2 = 1.5  # Reduced social component for less global influence
        self.mutation_rate = 0.1  # Reduced mutation rate for more stable exploitation
        self.crossover_rate = 0.6  # Slightly lower crossover rate

    def __call__(self, func):
        # Linearly reduce swarm size over time
        num_particles = self.init_particles
        positions = np.random.uniform(self.lb, self.ub, (num_particles, self.dim))
        velocities = np.random.uniform(-0.2, 0.2, (num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(num_particles, float('inf'))
        global_best_position = np.zeros(self.dim)
        global_best_score = float('inf')

        eval_count = 0

        while eval_count < self.budget:
            # Evaluate particles
            scores = np.array([func(p) for p in positions])
            eval_count += num_particles

            # Update personal and global bests
            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best_positions[better_mask] = positions[better_mask]

            min_score_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_score_idx] < global_best_score:
                global_best_score = personal_best_scores[min_score_idx]
                global_best_position = personal_best_positions[min_score_idx]

            # Adaptive cognitive and social components
            adaptive_c1 = self.c1 * np.exp(-eval_count / (0.5 * self.budget))
            adaptive_c2 = self.c2 * (1 - np.exp(-eval_count / (0.5 * self.budget)))

            # Update velocities and positions
            r1 = np.random.rand(num_particles, self.dim)
            r2 = np.random.rand(num_particles, self.dim)
            velocities = (self.w * velocities +
                          adaptive_c1 * r1 * (personal_best_positions - positions) +
                          adaptive_c2 * r2 * (global_best_position - positions))
            positions += velocities
            positions = np.clip(positions, self.lb, self.ub)

            # Genetic Algorithm operations
            for i in range(num_particles):
                if np.random.rand() < self.crossover_rate:
                    partner_idx = (i + np.random.randint(1, num_particles)) % num_particles
                    partner = positions[partner_idx]
                    mask = np.random.rand(self.dim) < 0.5
                    child = np.where(mask, positions[i], partner)

                    if np.random.rand() < self.mutation_rate:
                        mutation_idx = np.random.randint(self.dim)
                        mutation_value = np.random.uniform(self.lb, self.ub)
                        child[mutation_idx] = mutation_value

                    child_score = func(child)
                    eval_count += 1
                    if child_score < scores[i]:
                        positions[i] = child
                        scores[i] = child_score

                    if eval_count >= self.budget:
                        break

            # Update global best with the latest evaluations
            current_best_idx = np.argmin(scores)
            if scores[current_best_idx] < global_best_score:
                global_best_score = scores[current_best_idx]
                global_best_position = positions[current_best_idx]

            # Adjust swarm size
            num_particles = int(self.init_particles - (self.init_particles - self.final_particles) * (eval_count / self.budget))
            if num_particles < len(positions):
                positions = positions[:num_particles]
                velocities = velocities[:num_particles]
                personal_best_positions = personal_best_positions[:num_particles]
                personal_best_scores = personal_best_scores[:num_particles]

        return global_best_position