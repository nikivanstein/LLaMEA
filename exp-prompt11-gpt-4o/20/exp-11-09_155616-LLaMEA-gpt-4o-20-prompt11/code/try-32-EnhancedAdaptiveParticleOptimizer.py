import numpy as np

class EnhancedAdaptiveParticleOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, self.budget // 3)  # Slightly increased for diversity
        self.initial_inertia_weight = 0.7  # Start with a higher inertia weight
        self.final_inertia_weight = 0.4
        self.cognitive_coeff = 1.8  # Adjusted for better personal exploration
        self.social_coeff = 1.6
        self.dynamic_scaling = 0.95  # Introduced dynamic scaling for velocities

    def __call__(self, func):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
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

        while eval_count < self.budget:
            inertia_weight = (self.final_inertia_weight + 
                              (self.initial_inertia_weight - self.final_inertia_weight) * 
                              (self.budget - eval_count) / self.budget)
            
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (personal_best_positions - positions)
            social_component = self.social_coeff * r2 * (global_best_position - positions)
            velocities = inertia_weight * velocities + cognitive_component + social_component
            velocities *= self.dynamic_scaling  # Dynamic velocity scaling
            positions += velocities
            
            if eval_count % (self.budget // 4) == 0:  # More frequent adaptive adjustment
                median_score = np.median(scores)
                self.dynamic_scaling *= 0.98 if median_score > global_best_score else 1.02
                top_elite_index = np.argmin(scores)
                elite_position = positions[top_elite_index]
                positions = positions + (elite_position - positions) * 0.5
            
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