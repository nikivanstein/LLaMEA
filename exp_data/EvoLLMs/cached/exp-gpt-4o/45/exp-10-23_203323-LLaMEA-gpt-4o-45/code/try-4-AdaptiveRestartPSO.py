import numpy as np

class AdaptiveRestartPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.restart_threshold = 0.2 * dim
    
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
                        stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                if evaluations >= self.budget:
                    break

            inertia_weight = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)
            
            for i in range(self.num_particles):
                cognitive_component = self.c1 * np.random.uniform(0, 1, self.dim) * (personal_best_positions[i] - positions[i])
                social_component = self.c2 * np.random.uniform(0, 1, self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)
            
            if stagnation_counter > self.restart_threshold:
                positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
                velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
                stagnation_counter = 0

# Usage:
# adaptive_restart_pso = AdaptiveRestartPSO(budget=10000, dim=10)
# adaptive_restart_pso(func)