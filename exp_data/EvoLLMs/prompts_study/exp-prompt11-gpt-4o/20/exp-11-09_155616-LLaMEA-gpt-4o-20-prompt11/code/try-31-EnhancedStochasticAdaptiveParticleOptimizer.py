import numpy as np

class EnhancedStochasticAdaptiveParticleOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, self.budget // 2)  # Slight increase in population size
        self.inertia_weight = 0.5  # Further reduced for faster convergence
        self.cognitive_coeff = 1.8  # Slightly reduced for balance
        self.social_coeff = 1.7  # Slightly increased for enhanced social influence
        self.adaptive_scaling = 0.85  # Adjusted for better scaling dynamics

    def __call__(self, func):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))  # Narrower velocity range
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
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            r3 = np.random.rand(self.population_size, self.dim)  # Additional random component for perturbation
            cognitive_component = self.cognitive_coeff * r1 * (personal_best_positions - positions)
            social_component = self.social_coeff * r2 * (global_best_position - positions)
            perturbation = 0.1 * r3 * (np.random.randn(*positions.shape))  # Strategic noise
            velocities = self.inertia_weight * velocities + cognitive_component + social_component + perturbation
            positions += velocities
            
            if eval_count % (self.budget // 4) == 0:  # More frequent adaptive adjustment
                median_score = np.median(scores)
                self.inertia_weight *= self.adaptive_scaling if median_score > global_best_score else 1.0 / self.adaptive_scaling
                top_elite_index = np.argmin(scores)
                elite_position = positions[top_elite_index]
                positions = (positions + elite_position) / 2
            
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