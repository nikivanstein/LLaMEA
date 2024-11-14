import numpy as np

class AMPSO:
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
    
    def __call__(self, func):
        # Initialize particles
        positions = np.random.uniform(self.pos_bounds[0], self.pos_bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(self.vel_bounds[0], self.vel_bounds[1], (self.population_size, self.dim))
        
        pbest_positions = positions.copy()
        pbest_scores = np.array([func(pos) for pos in pbest_positions])
        gbest_position = pbest_positions[np.argmin(pbest_scores)]
        gbest_score = np.min(pbest_scores)

        for _ in range(self.iterations):
            # Adaptive inertia weight
            inertia_weight = self.inertia * (1 - self.memory_rate * (_ / self.iterations))

            # Update velocities
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (inertia_weight * velocities +
                          self.cognitive_coef * r1 * (pbest_positions - positions) +
                          self.social_coef * r2 * (gbest_position - positions))
            velocities = np.clip(velocities, self.vel_bounds[0], self.vel_bounds[1])

            # Update positions
            positions += velocities
            positions = np.clip(positions, self.pos_bounds[0], self.pos_bounds[1])

            # Evaluate function
            scores = np.array([func(pos) for pos in positions])

            # Update personal bests
            better_indices = scores < pbest_scores
            pbest_positions[better_indices] = positions[better_indices]
            pbest_scores[better_indices] = scores[better_indices]

            # Update global best
            if np.min(pbest_scores) < gbest_score:
                gbest_score = np.min(pbest_scores)
                gbest_position = pbest_positions[np.argmin(pbest_scores)]

        return gbest_position