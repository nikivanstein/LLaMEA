import numpy as np

class EnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(40, self.budget // 2)  # Increased population size
        self.inertia_weight = 0.7  # Adjusted inertia weight for exploration balance
        self.cognitive_coeff = 1.4  
        self.social_coeff = 1.6  # Slightly increased social coefficient
        self.adaptive_scaling = 0.95  # Adjusted adaptive scaling
        self.velocity_decay = 0.98  # New velocity decay factor

    def __call__(self, func):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-0.3, 0.3, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)
        
        scores = np.array([func(pos) for pos in positions])
        eval_count = self.population_size
        
        global_best_index = np.argmin(scores)
        global_best_position = np.copy(personal_best_positions[global_best_index])
        global_best_score = scores[global_best_index]
        
        personal_better_mask = scores < personal_best_scores
        personal_best_scores[personal_better_mask] = scores[personal_better_mask]
        personal_best_positions[personal_better_mask] = positions[personal_better_mask]

        velocity_clamp = (self.upper_bound - self.lower_bound) * 0.1  # Updated dynamic velocity clamp

        while eval_count < self.budget:
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (personal_best_positions - positions)
            social_component = self.social_coeff * r2 * (global_best_position - positions)
            velocities = self.velocity_decay * (self.inertia_weight * velocities + cognitive_component + social_component)
            
            velocities = np.clip(velocities, -velocity_clamp, velocity_clamp)
            positions += velocities
            
            if eval_count % (self.budget // 3) == 0:  # More frequent adaptive adjustment
                self.inertia_weight *= self.adaptive_scaling
                perturbation = np.random.normal(0, 0.03, positions.shape)  # Lower perturbation
                positions += perturbation
            
            positions = np.clip(positions, self.lower_bound, self.upper_bound)

            scores = np.array([func(pos) for pos in positions])
            eval_count += self.population_size
            
            personal_better_mask = scores < personal_best_scores
            personal_best_scores[personal_better_mask] = scores[personal_better_mask]
            personal_best_positions[personal_better_mask] = positions[personal_better_mask]
            
            current_global_best_index = np.argmin(personal_best_scores)
            current_global_best_score = personal_best_scores[current_global_best_index]
            if current_global_best_score < global_best_score:
                global_best_score = current_global_best_score
                global_best_position = np.copy(personal_best_positions[current_global_best_index])
                
        return global_best_score, global_best_position