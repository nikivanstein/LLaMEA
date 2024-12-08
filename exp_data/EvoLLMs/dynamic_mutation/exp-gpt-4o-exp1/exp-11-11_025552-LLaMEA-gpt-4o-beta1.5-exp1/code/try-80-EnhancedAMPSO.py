import numpy as np

class EnhancedAMPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.vel_bounds = (-0.5, 0.5)
        self.pos_bounds = (-5.0, 5.0)
        self.memory_rate = 0.1
        self.iterations = self.budget // self.population_size

    def oppositional_learning(self, positions):
        opposite_positions = self.pos_bounds[0] + self.pos_bounds[1] - positions
        return opposite_positions

    def adaptive_topology(self, positions, scores):
        neighborhood_size = max(1, int(self.population_size * 0.1))
        indices = np.argsort(scores)
        return positions[indices[:neighborhood_size]], scores[indices[:neighborhood_size]]

    def dynamic_population_size(self, iteration, max_iterations):
        # Dynamic adjustment of population size
        return int(self.population_size * (1 - iteration / max_iterations))

    def dimension_reduction(self, positions, iteration):
        # Reduce dimensionality over iterations
        if iteration % 10 == 0:
            cutoff_dim = max(1, int(self.dim * (1 - iteration / self.iterations)))
            positions[:, cutoff_dim:] = 0
        return positions

    def __call__(self, func):
        positions = np.random.uniform(self.pos_bounds[0], self.pos_bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(self.vel_bounds[0], self.vel_bounds[1], (self.population_size, self.dim))
        
        opposite_positions = self.oppositional_learning(positions)
        oppositional_scores = np.array([func(pos) for pos in opposite_positions])
        scores = np.array([func(pos) for pos in positions])
        
        better_initial = oppositional_scores < scores
        positions[better_initial] = opposite_positions[better_initial]
        scores[better_initial] = oppositional_scores[better_initial]

        pbest_positions = positions.copy()
        pbest_scores = scores.copy()
        gbest_position = pbest_positions[np.argmin(pbest_scores)]
        gbest_score = np.min(pbest_scores)

        for i in range(self.iterations):
            self.memory_rate = 0.1 + 0.9 * (i / self.iterations)
            inertia_weight = self.inertia * (1 - (i / self.iterations)**2)

            self.cognitive_coef = 1.5 + 0.5 * (i / self.iterations)

            current_population_size = self.dynamic_population_size(i, self.iterations)

            local_positions, local_scores = self.adaptive_topology(positions[:current_population_size], scores[:current_population_size])
            local_best_position = local_positions[np.argmin(local_scores)]

            r1, r2 = np.random.rand(current_population_size, self.dim), np.random.rand(current_population_size, self.dim)
            velocities = (inertia_weight * velocities[:current_population_size] +
                          self.cognitive_coef * r1 * (pbest_positions[:current_population_size] - positions[:current_population_size]) +
                          self.social_coef * r2 * (local_best_position - positions[:current_population_size]))
            velocities = np.clip(velocities, self.vel_bounds[0], self.vel_bounds[1])

            positions[:current_population_size] += velocities
            positions[:current_population_size] = np.clip(positions[:current_population_size], self.pos_bounds[0], self.pos_bounds[1])

            positions = self.dimension_reduction(positions, i)

            scores[:current_population_size] = np.array([func(pos) for pos in positions[:current_population_size]])

            better_indices = scores[:current_population_size] < pbest_scores[:current_population_size]
            pbest_positions[:current_population_size][better_indices] = positions[:current_population_size][better_indices]
            pbest_scores[:current_population_size][better_indices] = scores[:current_population_size][better_indices]

            if np.min(pbest_scores) < gbest_score:
                gbest_score = np.min(pbest_scores)
                gbest_position = pbest_positions[np.argmin(pbest_scores)]

        return gbest_position