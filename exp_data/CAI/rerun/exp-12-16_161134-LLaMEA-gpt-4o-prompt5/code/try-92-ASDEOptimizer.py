import numpy as np

class ASDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0

    def __call__(self, func):
        population_size = int(min(50, self.budget // 10))
        cognitive_coeff_base = 1.7  # Changed from 1.5 to 1.7 for enhanced exploration
        social_coeff_base = 1.5

        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.full(population_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        while self.evaluations < self.budget:
            for i in range(population_size):
                if self.evaluations >= self.budget:
                    break
                score = func(swarm[i])
                self.evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i]
                    
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = swarm[i]

            # Update velocities and positions
            for i in range(population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                inertia_weight = 0.95 - 0.4 * (self.evaluations / self.budget)  # Increased initial inertia weight
                
                decay_factor = 0.96 + 0.02 * (self.evaluations / self.budget)  # Enhanced adaptive decay factor
                cognitive_coeff = cognitive_coeff_base * (1 - self.evaluations / self.budget) * decay_factor
                social_coeff = social_coeff_base * (self.evaluations / self.budget) * decay_factor

                velocities[i] = (inertia_weight * velocities[i] +
                                 cognitive_coeff * r1 * (personal_best_positions[i] - swarm[i]) +
                                 social_coeff * r2 * (global_best_position - swarm[i]))

                # Slightly increased adaptive velocity clamping factor
                max_velocity = 0.55 * np.linalg.norm(global_best_position)

                # Modified line for velocity clamping
                velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)

                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)

            # Dynamic population resizing
            population_size = int(min(50, self.budget // 10) * (0.5 + 0.5 * self.evaluations / self.budget))

        return {"best_position": global_best_position, "best_score": global_best_score}