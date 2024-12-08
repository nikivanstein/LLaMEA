import numpy as np

class AdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 + 5 * self.dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia = 0.7  # Inertia weight for velocity update
        self.cognitive_coeff = 1.5  # Cognitive (personal) coefficient
        self.social_coeff = 1.5  # Social (global) coefficient

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.zeros_like(positions)
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.array([func(ind) for ind in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)
        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                # Update velocity with inertia, cognitive, and social components
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_coeff * r2 * (global_best_position - positions[i]))
                
                # Update position
                positions[i] = np.clip(positions[i] + velocities[i], self.lower_bound, self.upper_bound)

                # Evaluate fitness
                fitness = func(positions[i])
                evaluations += 1

                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = fitness

                # Update global best
                if fitness < global_best_fitness:
                    global_best_position = positions[i]
                    global_best_fitness = fitness

        return global_best_position