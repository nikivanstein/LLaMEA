import numpy as np

class HybridPSOGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 40
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_initial = 2.5
        self.c2_initial = 1.5
        self.c1_final = 1.5
        self.c2_final = 2.5
        self.mutation_rate = 0.1  # Mutation rate for GA component
        self.velocity_clamp = 0.6  # Increased velocity clamping for more control
    
    def __call__(self, func):
        np.random.seed(0)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))  # Start with zero velocities for stability
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
            c1 = self.c1_initial - ((self.c1_initial - self.c1_final) * evaluations / self.budget)
            c2 = self.c2_initial + ((self.c2_final - self.c2_initial) * evaluations / self.budget)
            
            for i in range(self.num_particles):
                cognitive_component = c1 * np.random.uniform(0, 1, self.dim) * (personal_best_positions[i] - positions[i])
                social_component = c2 * np.random.uniform(0, 1, self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                
                # Clamping velocities for stability
                velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)
                
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

            # Genetic Algorithm-inspired crossover
            for i in range(0, self.num_particles, 2):
                if i+1 < self.num_particles:
                    alpha = np.random.rand(self.dim)
                    child1 = alpha * positions[i] + (1 - alpha) * positions[i+1]
                    child2 = alpha * positions[i+1] + (1 - alpha) * positions[i]
                    positions[i], positions[i+1] = child1, child2
                    # Mutation
                    if np.random.rand() < self.mutation_rate:
                        mutation_vector = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                        positions[i] = mutation_vector
                    if np.random.rand() < self.mutation_rate:
                        mutation_vector = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                        positions[i+1] = mutation_vector

# Usage:
# hybrid_pso_ga = HybridPSOGA(budget=10000, dim=10)
# hybrid_pso_ga(func)