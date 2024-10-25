import numpy as np

class EnhancedPSOWithSwarmRegeneration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.9
        self.w_min = 0.2  # Modified for wider dynamic range
        self.c1_initial = 2.0
        self.c2_initial = 2.0
        self.regeneration_threshold = 0.3 * self.budget  # Allowing for swarm regeneration if stuck
    
    def __call__(self, func):
        np.random.seed(0)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.num_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        stagnation_counter = 0

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
                        stagnation_counter = 0  # Reset stagnation counter
                else:
                    stagnation_counter += 1
                
                if evaluations >= self.budget:
                    break

            if stagnation_counter > self.regeneration_threshold:
                # Swarm regeneration strategy
                positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
                velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
                stagnation_counter = 0

            inertia_weight = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)
            c1 = self.c1_initial
            c2 = self.c2_initial
            
            for i in range(self.num_particles):
                cognitive_component = c1 * np.random.uniform(0, 1, self.dim) * (personal_best_positions[i] - positions[i])
                social_component = c2 * np.random.uniform(0, 1, self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component

                # Remove velocity clamping for more varied exploration
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

# Usage:
# enhanced_pso = EnhancedPSOWithSwarmRegeneration(budget=10000, dim=10)
# enhanced_pso(func)