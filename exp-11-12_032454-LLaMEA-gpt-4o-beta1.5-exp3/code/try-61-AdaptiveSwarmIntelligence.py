import numpy as np

class AdaptiveSwarmIntelligence:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.swarm_size = max(5, int(budget / (10 * dim)))
        self.c1_initial = 2.0  # Cognitive coefficient
        self.c2_initial = 2.0  # Social coefficient
        self.c1_final = 0.5    # Final cognitive coefficient
        self.c2_final = 0.5    # Final social coefficient
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4

    def __call__(self, func):
        # Initialize swarm
        positions = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.array([func(p) for p in positions])
        num_evaluations = self.swarm_size

        # Find initial global best
        best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[best_idx]
        global_best_fitness = personal_best_fitness[best_idx]

        while num_evaluations < self.budget:
            # Dynamic inertia weight and coefficients
            inertia_weight = self.inertia_weight_initial - (self.inertia_weight_initial - self.inertia_weight_final) * (num_evaluations / self.budget)
            c1 = self.c1_initial - (self.c1_initial - self.c1_final) * (num_evaluations / self.budget)
            c2 = self.c2_initial - (self.c2_initial - self.c2_final) * (num_evaluations / self.budget)

            for i in range(self.swarm_size):
                if num_evaluations >= self.budget:
                    break

                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = c1 * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = c2 * r2 * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_velocity + social_velocity

                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lb, self.ub)

                # Evaluate new fitness
                fitness = func(positions[i])
                num_evaluations += 1

                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = fitness

                # Update global best
                if fitness < global_best_fitness:
                    global_best_position = positions[i]
                    global_best_fitness = fitness

        return global_best_position, global_best_fitness