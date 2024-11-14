import numpy as np

class EnhancedParticleSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, self.budget // 2)
        self.inertia_weight = 0.5  # Further decreased inertia for more exploration
        self.cognitive_coeff = 1.7  # Slightly increased for improved personal exploration
        self.social_coeff = 1.3  # Slightly decreased to reduce premature convergence
        self.adaptive_scaling = 0.92  # Adjusted scaling for robust adaptation

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

        velocity_clamp = (self.upper_bound - self.lower_bound) * 0.1  # Reduced dynamic velocity clamp

        while eval_count < self.budget:
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (personal_best_positions - positions)
            social_component = self.social_coeff * r2 * (global_best_position - positions)
            velocities = self.inertia_weight * velocities + cognitive_component + social_component
            
            velocities = np.clip(velocities, -velocity_clamp, velocity_clamp)
            positions += velocities
            
            if eval_count % (self.budget // 3) == 0:  # Adjusted frequency for adaptive adjustment
                self.inertia_weight *= self.adaptive_scaling
                perturbation = np.random.uniform(-0.05, 0.05, positions.shape)  # Uniform perturbation for exploration
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

            # Local search enhancement step
            if eval_count % (self.budget // 5) == 0:
                local_search_idx = np.random.randint(self.population_size)
                candidate = positions[local_search_idx] + np.random.normal(0, 0.1, self.dim)
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_score = func(candidate)
                eval_count += 1
                if candidate_score < personal_best_scores[local_search_idx]:
                    personal_best_scores[local_search_idx] = candidate_score
                    personal_best_positions[local_search_idx] = candidate
                    if candidate_score < global_best_score:
                        global_best_score = candidate_score
                        global_best_position = candidate
                
        return global_best_score, global_best_position