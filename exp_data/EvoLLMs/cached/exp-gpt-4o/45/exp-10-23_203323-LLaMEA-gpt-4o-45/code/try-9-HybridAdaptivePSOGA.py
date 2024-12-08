import numpy as np

class HybridAdaptivePSOGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_initial = 2.5
        self.c2_initial = 1.5
        self.c1_final = 1.5
        self.c2_final = 2.5
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
    
    def __call__(self, func):
        np.random.seed(0)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.num_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        while evaluations < self.budget:
            # Evaluate and update personal and global bests
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

            # Adaptive inertia weight and coefficients
            inertia_weight = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)
            c1 = self.c1_initial - ((self.c1_initial - self.c1_final) * evaluations / self.budget)
            c2 = self.c2_initial + ((self.c2_final - self.c2_initial) * evaluations / self.budget)
            
            # Particle update with crossover and mutation
            for i in range(self.num_particles):
                cognitive_component = c1 * np.random.uniform(0, 1, self.dim) * (personal_best_positions[i] - positions[i])
                social_component = c2 * np.random.uniform(0, 1, self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                
                # Crossover
                if np.random.rand() < self.crossover_rate:
                    partner_idx = np.random.randint(self.num_particles)
                    cross_point = np.random.randint(self.dim)
                    positions[i][:cross_point] = personal_best_positions[partner_idx][:cross_point]
                
                # Mutation
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    positions[i] += mutation_vector * np.random.uniform(-1, 1, self.dim)
                
                # Update position
                positions[i] = np.clip(positions[i] + velocities[i], self.lower_bound, self.upper_bound)

# Usage:
# hybrid_pso_ga = HybridAdaptivePSOGA(budget=10000, dim=10)
# hybrid_pso_ga(func)