import numpy as np

class NovelPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.inertia_weight = 0.729  # inertia weight
        self.cognitive_const = 1.49445  # cognitive constant
        self.social_const = 1.49445  # social constant
        self.particles = np.random.uniform(-5, 5, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            # Evaluate the fitness of each particle
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                
                score = func(self.particles[i])
                self.evaluations += 1

                # Update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]

            # Update velocities and positions
            for i in range(self.population_size):
                cognitive_component = (self.cognitive_const * 
                                       np.random.rand(self.dim) * 
                                       (self.personal_best_positions[i] - self.particles[i]))
                social_component = (self.social_const * 
                                    np.random.rand(self.dim) * 
                                    (self.global_best_position - self.particles[i]))

                self.velocities[i] = (
                    self.inertia_weight * self.velocities[i] + 
                    cognitive_component + 
                    social_component
                )

                # Ensure velocities are within bounds
                self.velocities[i] = np.clip(self.velocities[i], -1, 1)

                # Update particle positions
                self.particles[i] += self.velocities[i]

                # Ensure particles are within bounds
                self.particles[i] = np.clip(self.particles[i], -5, 5)

        # Return the best found solution
        return self.global_best_position, self.global_best_score