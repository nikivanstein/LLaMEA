import numpy as np

class EnhancedDynamicPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.9
        self.w_min = 0.3  # Modified for improved exploration
        self.c1_initial = 2.8  # Adjusted for stronger cognitive influence initially
        self.c2_initial = 1.2  # Adjusted for reduced social influence initially
        self.c1_final = 1.2  # Adjusted for reduced cognitive influence towards final
        self.c2_final = 2.8  # Adjusted for stronger social influence towards final
    
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

            inertia_weight = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget) ** 1.2)  # Non-linear decay
            c1 = self.c1_initial - ((self.c1_initial - self.c1_final) * (evaluations / self.budget))
            c2 = self.c2_initial + ((self.c2_final - self.c2_initial) * (evaluations / self.budget))
            
            for i in range(self.num_particles):
                cognitive_component = c1 * np.random.uniform(0, 1, self.dim) * (personal_best_positions[i] - positions[i])
                social_component = c2 * np.random.uniform(0, 1, self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

# Usage:
# enhanced_dynamic_pso = EnhancedDynamicPSO(budget=10000, dim=10)
# enhanced_dynamic_pso(func)