import numpy as np

class EnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.5  # Adjusted cognitive coefficient
        self.c2 = 2.5  # Adjusted social coefficient
        self.neighborhood_size = 5  # Neighborhood size for local best
        self.alpha = 0.5  # Learning rate adaptation factor
    
    def __call__(self, func):
        np.random.seed(0)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.num_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        local_best_positions = personal_best_positions.copy()

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
                # Determine local best in neighborhood
                start_idx = max(0, i - self.neighborhood_size // 2)
                end_idx = min(self.num_particles, i + self.neighborhood_size // 2)
                local_best_score = float('inf')
                for j in range(start_idx, end_idx):
                    if personal_best_scores[j] < local_best_score:
                        local_best_score = personal_best_scores[j]
                        local_best_positions[i] = personal_best_positions[j].copy()
                
                # Dynamic learning rate adjustment
                dynamic_c1 = self.c1 * (1 + self.alpha * np.random.uniform(-0.5, 0.5))
                dynamic_c2 = self.c2 * (1 + self.alpha * np.random.uniform(-0.5, 0.5))

                cognitive_component = dynamic_c1 * np.random.uniform(0, 1, self.dim) * (personal_best_positions[i] - positions[i])
                social_component = dynamic_c2 * np.random.uniform(0, 1, self.dim) * (local_best_positions[i] - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

# Usage:
# pso = EnhancedAdaptivePSO(budget=10000, dim=10)
# pso(func)