import numpy as np

class DynamicNeighborhoodPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 60  # Increased the number of particles for enhanced exploration
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.8
        self.w_min = 0.4  # Adjusted minimum inertia weight for dynamic adaptation
        self.c1 = 1.5  # Balanced cognitive factor for individual insight
        self.c2 = 2.0  # Moderate social factor for collaborative guidance
        self.velocity_clamp = 0.4  # Refined velocity clamping to control oscillations
        self.neighborhood_size = 5  # Introduced neighborhood-based interactions
        self.adaptive_threshold = 0.05  # Threshold for adaptively adjusting exploration

    def __call__(self, func):
        np.random.seed(0)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.num_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.num_particles):
                score = func(positions[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = positions[i].copy()

                if evaluations >= self.budget:
                    break

            inertia_weight = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)

            for i in range(self.num_particles):
                # Dynamic neighborhood interaction
                neighbors = np.random.choice(self.num_particles, self.neighborhood_size, replace=False)
                local_best_position = min(neighbors, key=lambda idx: personal_best_scores[idx])
                local_best_position = personal_best_positions[local_best_position]

                if np.random.rand() < self.adaptive_threshold:
                    perturbation = np.random.normal(0, 0.1, self.dim)
                    candidate_position = positions[i] + perturbation
                    candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
                    candidate_score = func(candidate_position)
                    evaluations += 1
                    if candidate_score < personal_best_scores[i]:
                        personal_best_scores[i] = candidate_score
                        personal_best_positions[i] = candidate_position.copy()
                        if candidate_score < global_best_score:
                            global_best_score = candidate_score
                            global_best_position = candidate_position.copy()

                cognitive_component = self.c1 * np.random.uniform(0, 1, self.dim) * (personal_best_positions[i] - positions[i])
                social_component = self.c2 * np.random.uniform(0, 1, self.dim) * (local_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                
                velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)
                
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

# Usage:
# dynamic_neighborhood_pso = DynamicNeighborhoodPSO(budget=10000, dim=10)
# dynamic_neighborhood_pso(func)