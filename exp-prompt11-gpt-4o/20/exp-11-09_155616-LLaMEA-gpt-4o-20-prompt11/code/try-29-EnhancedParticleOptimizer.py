import numpy as np

class EnhancedParticleOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, self.budget // 2)  # Increase population size
        self.inertia_weight = 0.9  # Start with higher inertia for exploration
        self.cognitive_coeff = 1.5  # Adjusted to reduce personal bias
        self.social_coeff = 1.7  # Slightly increased for global influence
        self.adaptive_scaling = 0.8  # More aggressive scaling
        self.mutation_prob = 0.1  # Introduce mutation probability

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
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (personal_best_positions - positions)
            social_component = self.social_coeff * r2 * (global_best_position - positions)
            velocities = self.inertia_weight * velocities + cognitive_component + social_component
            positions += velocities
            
            mutation_mask = np.random.rand(self.population_size, self.dim) < self.mutation_prob
            random_mutation = np.random.uniform(-1, 1, (self.population_size, self.dim))
            positions = np.where(mutation_mask, positions + random_mutation, positions)
            
            if eval_count % (self.budget // 5) == 0:  # Adjust inertia dynamically
                median_score = np.median(scores)
                self.inertia_weight *= self.adaptive_scaling if median_score > global_best_score else 1.1
                elite_position = positions[np.argmin(scores)]
                positions = (positions + 2 * elite_position) / 3  # Refined elite influence
            
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