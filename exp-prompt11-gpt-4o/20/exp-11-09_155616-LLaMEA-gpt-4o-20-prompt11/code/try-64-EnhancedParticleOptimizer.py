import numpy as np

class EnhancedParticleOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, self.budget // 3)  # Increased initial diversity
        self.inertia_weight = 0.8  # Adaptive inertia for better exploration-exploitation balance
        self.cognitive_coeff = 2.0  # Enhanced cognitive component for individual learning
        self.social_coeff = 1.3  # Adjusted social factor for collective learning
        self.adaptive_scaling = 0.95  # Smoother adaptive scaling

    def __call__(self, func):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))  # Broader initial velocity range
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

        velocity_clamp = (self.upper_bound - self.lower_bound) * 0.20  # Dynamic velocity clamp heightened

        while eval_count < self.budget:
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (personal_best_positions - positions)
            social_component = self.social_coeff * r2 * (global_best_position - positions)
            velocities = self.inertia_weight * velocities + cognitive_component + social_component
            
            velocities = np.clip(velocities, -velocity_clamp, velocity_clamp)
            positions += velocities
            
            if eval_count % (self.budget // 8) == 0:  # More frequent inertia adjustment
                self.inertia_weight *= self.adaptive_scaling
                mutation = np.random.normal(0, 0.1, positions.shape)  # Slightly increased mutation for exploration
                positions += mutation
            
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