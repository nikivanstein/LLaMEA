import numpy as np

class AdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.inertia_weight = 0.9  # Increased initial inertia weight
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.vel_max = 0.1 * (self.upper_bound - self.lower_bound)
        self.vel_min = -self.vel_max
        
        # Initialize particles' positions and velocities
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(self.vel_min, self.vel_max, (self.population_size, self.dim))
        
        # Initialize personal bests and global best
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Evaluate the fitness of the particle
                score = func(self.positions[i])
                evaluations += 1

                # Update personal best
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i]
                
                # Update global best
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.positions[i]
        
            # Update velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = (self.cognitive_weight + np.random.rand() * 0.5) * r1 * (self.pbest_positions[i] - self.positions[i])  # Added stochastic component
                social_component = (self.social_weight + 0.5 * np.random.rand()) * r2 * (self.gbest_position - self.positions[i]) # Adaptive social weight
                self.velocities[i] = (self.inertia_weight * self.velocities[i] + 
                                      cognitive_component + social_component)
                
                # Clamp velocities
                self.velocities[i] = np.clip(self.velocities[i], self.vel_min, self.vel_max)
                
                # Update position
                self.positions[i] += self.velocities[i]
                
                # Clamp positions
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Dynamically reduce inertia weight
            self.inertia_weight *= (0.99 + 0.01 * np.random.rand()) * (1 - evaluations / self.budget)  # Temperature-based inertia reduction

        return self.gbest_score, self.gbest_position